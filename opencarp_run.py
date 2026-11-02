"""Helpers shared by the examples: run openCARP on files written by io4dolfinx and map
the activation times it computes back onto a mesh read with io4dolfinx."""

import os
import shutil
import subprocess
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np

import io4dolfinx

# conda-forge openCARP in this container, used if openCARP is not on PATH
_default_opencarp = "/opt/micromamba/envs/opencarp/bin/openCARP"


def opencarp_executable() -> str:
    """openCARP from $OPENCARP, else from PATH, else the conda-forge install."""
    exe = os.environ.get("OPENCARP") or shutil.which("openCARP") or _default_opencarp
    if not Path(exe).exists():
        raise FileNotFoundError("Could not find openCARP, set $OPENCARP to its path.")
    return exe


def write_parameters(path: Path, parameters: dict):
    """Write an openCARP parameter file, one ``name = value`` line per parameter."""
    lines = [f"{name} = {value}" for name, value in parameters.items()]
    path.write_text("\n".join(lines) + "\n")


def run_opencarp(outdir: Path, parameters: dict, comm: MPI.Comm = MPI.COMM_WORLD) -> Path:
    """Run openCARP in ``outdir`` on rank 0 with the given parameters.

    Returns:
        The output directory of the simulation, ``outdir / parameters["simID"]``
    """
    par = outdir / "simulation.par"
    if comm.rank == 0:
        write_parameters(par, parameters)
        log = outdir / "opencarp.log"
        # openCARP is an MPI program: started from a process launched by mpirun, it
        # would try to join that job, so drop the variables of the MPI launcher
        launcher = ("PMI_", "PMIX_", "HYDRA_", "MPIR_", "OMPI_")
        env = {k: v for k, v in os.environ.items() if not k.startswith(launcher)}
        with open(log, "w") as f:
            result = subprocess.run(
                [opencarp_executable(), "+F", par.name],
                cwd=outdir,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise RuntimeError(f"openCARP failed, see {log}:\n{log.read_text()[-3000:]}")
        print(f"openCARP finished, log in {log}")
    comm.Barrier()
    return outdir / parameters["simID"]


def activation_function(
    mesh: dolfinx.mesh.Mesh, activation_file: Path, name: str = "activation"
) -> dolfinx.fem.Function:
    """Read activation times per openCARP node into a P-1 function on a mesh read
    with io4dolfinx from the same files.

    Nodes that never activate get ``-1`` from openCARP.
    """
    times = np.loadtxt(activation_file, ndmin=1)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V, name=name)
    # The geometry node i of a mesh read with io4dolfinx is node
    # input_global_indices[i] of the .pts file, and P-1 dofs follow the geometry nodes
    nodes = mesh.geometry.input_global_indices[mesh.geometry.dofmap]
    u.x.array[V.dofmap.list] = times[nodes]
    return u


def owned_values(u: dolfinx.fem.Function) -> tuple[np.ndarray, np.ndarray] | None:
    """Gather the dof coordinates and values of a P-1 function to rank 0.

    Returns:
        The coordinates and values on rank 0, ``None`` on the other ranks
    """
    V = u.function_space
    num_owned = V.dofmap.index_map.size_local
    x = V.tabulate_dof_coordinates()[:num_owned]
    values = u.x.array[:num_owned]
    gathered = V.mesh.comm.gather((x, values), root=0)
    if gathered is None:
        return None
    return np.vstack([g[0] for g in gathered]), np.concatenate([g[1] for g in gathered])


def read_mesh_and_activation(base: Path, activation_file: Path, scale: float, xdmf: Path):
    """Read the mesh and its region tags back with io4dolfinx, add the activation
    times computed by openCARP, and save them for ParaView."""
    comm = MPI.COMM_WORLD
    args = {"scale": scale}
    elem = base.with_suffix(".elem")
    mesh = io4dolfinx.read_mesh(elem, comm, backend="opencarp", backend_args=args)
    regions = io4dolfinx.read_meshtags(elem, mesh, "cell", backend="opencarp")
    fiber = io4dolfinx.read_cell_data(elem, "fiber", mesh, backend="opencarp")
    activation = activation_function(mesh, activation_file)
    with dolfinx.io.XDMFFile(comm, xdmf, "w") as f:
        f.write_mesh(mesh)
        f.write_meshtags(regions, mesh.geometry)
        f.write_function(activation)
    print(f"Wrote {xdmf}")
    return mesh, regions, fiber, activation
