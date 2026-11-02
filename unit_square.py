"""Planar wave in a 1 cm x 1 cm square, stimulated on its left side.

Writes the mesh with io4dolfinx, runs openCARP on it, and checks the activation times:

- every node activates, and the wave is planar (activation depends on x only), which
  shows that the left-side ``.vtx`` marker and the fibers along x were read;
- the wave is slower in the right half (region 2), whose conductivities are a quarter
  of those of the left half (region 1), which shows that the region tags were read.

Run with ``python3 unit_square.py [--n 50] [--outdir unit_square]``.
"""

import argparse
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
from opencarp_run import owned_values, read_mesh_and_activation, run_opencarp

import io4dolfinx

# The mesh is in cm, openCARP wants µm
SCALE = 1e4
# Region of the left (x < 0.5) and right halves
LEFT, RIGHT = 1, 2


def write_mesh(base: Path, n: int):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, n, n, dolfinx.mesh.CellType.triangle)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    regions = np.where(midpoints[:, 0] < 0.5, LEFT, RIGHT).astype(np.int32)
    cell_tags = dolfinx.mesh.meshtags(mesh, tdim, cells, regions)

    left = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0))
    facet_tags = dolfinx.mesh.meshtags(mesh, tdim - 1, left, np.full_like(left, 1))

    V = dolfinx.fem.functionspace(mesh, ("DG", 0, (2,)))
    fiber = dolfinx.fem.Function(V, name="fiber")
    fiber.interpolate(lambda x: np.vstack([np.ones_like(x[0]), np.zeros_like(x[0])]))

    args = {"scale": SCALE, "markers": {"left": (1, tdim - 1)}}
    io4dolfinx.write_mesh_input_order(base, mesh, backend="opencarp", backend_args=args)
    for tags in (cell_tags, facet_tags):
        io4dolfinx.write_meshtags(
            base, mesh, tags, backend="opencarp", backend_args=args, on_input_mesh=True
        )
    io4dolfinx.write_cell_data(
        base, fiber, 0.0, io4dolfinx.FileMode.write, args, "opencarp", on_input_mesh=True
    )


def parameters(base: Path) -> dict:
    return {
        "meshname": base.name,
        "simID": "simulation",
        "bidomain": 0,
        "tend": 120.0,
        "dt": 25.0,
        "spacedt": 1.0,
        # Intracellular domain (monodomain) on both regions
        "num_phys_regions": 1,
        "phys_region[0].ptype": 0,
        "phys_region[0].num_IDs": 2,
        "phys_region[0].ID[0]": LEFT,
        "phys_region[0].ID[1]": RIGHT,
        # Ionic model in all regions
        "num_imp_regions": 1,
        "imp_region[0].im": "tenTusscherPanfilov",
        # Full conductivities on the left, a quarter on the right
        "num_gregions": 2,
        "gregion[0].num_IDs": 1,
        "gregion[0].ID[0]": LEFT,
        "gregion[1].num_IDs": 1,
        "gregion[1].ID[0]": RIGHT,
        "gregion[1].g_mult": 0.25,
        # Stimulus at the nodes of the left side
        "num_stim": 1,
        "stim[0].crct.type": 0,
        "stim[0].elec.vtx_file": f"{base.name}_left",
        "stim[0].ptcl.start": 0.0,
        "stim[0].ptcl.duration": 2.0,
        "stim[0].pulse.strength": 100.0,
        # Activation time per node: first upstroke through -10 mV
        "num_LATs": 1,
        "lats[0].ID": "activation",
        "lats[0].measurand": 0,
        "lats[0].all": 0,
        "lats[0].mode": 0,
        "lats[0].threshold": -10.0,
        "lats[0].method": 1,
    }


def check(activation: dolfinx.fem.Function):
    """Check the activation times on rank 0."""
    if (values := owned_values(activation)) is None:
        return
    x, t = values
    assert np.all(t >= 0), f"{np.count_nonzero(t < 0)} nodes did not activate"

    # Planar: the spread over y at fixed x is small compared to the total time
    columns = np.round(x[:, 0], 8)
    spread = max(np.ptp(t[columns == c]) for c in np.unique(columns))
    print(f"Activation from {t.min():.1f} to {t.max():.1f} ms, spread over y {spread:.2f} ms")
    assert spread < 0.05 * t.max(), "the wave is not planar"

    # Slower in the right half: conduction velocity from a fit of x against t
    def velocity(lo, hi):
        inside = (x[:, 0] > lo) & (x[:, 0] < hi)
        return np.polyfit(t[inside], x[inside, 0], 1)[0] * 10  # cm/ms to m/s

    v_left, v_right = velocity(0.1, 0.4), velocity(0.6, 0.9)
    print(f"Conduction velocity {v_left:.3f} m/s on the left, {v_right:.3f} m/s on the right")
    # The velocity scales roughly with the square root of the conductivity
    assert 0.35 < v_right / v_left < 0.65, "region 2 does not have its own conductivity"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=50, help="Cells per side")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "unit_square")
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    if MPI.COMM_WORLD.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()
    base = outdir / "square"

    write_mesh(base, args.n)
    simulation = run_opencarp(outdir, parameters(base))
    *_, activation = read_mesh_and_activation(
        base, simulation / "init_acts_activation-thresh.dat", SCALE, outdir / "activation.xdmf"
    )
    check(activation)
    if MPI.COMM_WORLD.rank == 0:
        print("unit square: OK")


if __name__ == "__main__":
    main()
