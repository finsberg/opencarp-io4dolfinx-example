"""The N-version benchmark of Niederer et al. (2011), doi:10.1098/rsta.2011.0139.

A 20 x 7 x 3 mm slab with fibers along x, the ten Tusscher-Panfilov (2006) model and a
stimulus in a 1.5 mm cube at the origin. The mesh, the fibers and sheets, and the
stimulus as a ``.vtx`` vertex marker are written with io4dolfinx, and openCARP
computes the activation time at every node.

The script prints the activation times at the benchmark points P1-P9 and checks that

- every node activates, and later at P8 (the far corner) than at P1 (the stimulus);
- conduction along x is faster than across, which shows that the fibers were read.

With ``--dx 0.2`` (the default, about 30 s) P8 activates after about 54 ms. The
converged value is about 42 ms, and coarser meshes activate later: at ``--dx 0.5``
conduction across the fibers is so slow that the far side of the slab does not
activate within the 100 ms simulated, and the check fails.

Run with ``python3 niederer.py [--dx 0.2] [--outdir niederer]``.
"""

import argparse
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
from opencarp_run import owned_values, read_mesh_and_activation, run_opencarp

import io4dolfinx

# The mesh is in mm, openCARP wants µm
SCALE = 1e3
SIZE = np.array([20.0, 7.0, 3.0])
STIMULUS = 1.5
STIMULUS_MARKER = 1

# Benchmark points, from the stimulus corner to the far corner, and the centre
POINTS = {
    "P1": (0, 0, 0),
    "P2": (0, 7, 0),
    "P3": (20, 0, 0),
    "P4": (20, 7, 0),
    "P5": (0, 0, 3),
    "P6": (0, 7, 3),
    "P7": (20, 0, 3),
    "P8": (20, 7, 3),
    "P9": (10, 3.5, 1.5),
}


def write_mesh(base: Path, dx: float):
    comm = MPI.COMM_WORLD
    n = [max(1, round(s / dx)) for s in SIZE]
    mesh = dolfinx.mesh.create_box(comm, [np.zeros(3), SIZE], n, dolfinx.mesh.CellType.tetrahedron)

    # Stimulus: the nodes in the cube [0, 1.5]^3 mm, as a vertex marker
    def in_stimulus(x):
        return np.all(x <= STIMULUS + 1e-8, axis=0)

    vertices = dolfinx.mesh.locate_entities(mesh, 0, in_stimulus)
    vertex_tags = dolfinx.mesh.meshtags(mesh, 0, vertices, np.full_like(vertices, STIMULUS_MARKER))

    V = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
    fiber = dolfinx.fem.Function(V, name="fiber")
    fiber.interpolate(lambda x: np.vstack([np.ones_like(x[0]), 0 * x[0], 0 * x[0]]))
    sheet = dolfinx.fem.Function(V, name="sheet")
    sheet.interpolate(lambda x: np.vstack([0 * x[0], np.ones_like(x[0]), 0 * x[0]]))

    args = {"scale": SCALE, "default_tag": 1, "markers": {"stimulus": (STIMULUS_MARKER, 0)}}
    io4dolfinx.write_mesh_input_order(base, mesh, backend="opencarp", backend_args=args)
    io4dolfinx.write_meshtags(
        base, mesh, vertex_tags, backend="opencarp", backend_args=args, on_input_mesh=True
    )
    for u, mode in ((fiber, io4dolfinx.FileMode.write), (sheet, io4dolfinx.FileMode.append)):
        io4dolfinx.write_cell_data(base, u, 0.0, mode, args, "opencarp", on_input_mesh=True)
    num_cells = mesh.topology.index_map(3).size_global
    if comm.rank == 0:
        print(f"Wrote a mesh with {num_cells} tetrahedra, {n} cells along the sides")


def parameters(base: Path) -> dict:
    return {
        "meshname": base.name,
        "simID": "simulation",
        "bidomain": 0,
        "tend": 100.0,
        "dt": 25.0,
        "spacedt": 1.0,
        "num_phys_regions": 1,
        "phys_region[0].ptype": 0,
        "phys_region[0].num_IDs": 1,
        "phys_region[0].ID[0]": 1,
        # Epicardial ten Tusscher-Panfilov, surface to volume ratio 1400 /cm
        "num_imp_regions": 1,
        "imp_region[0].im": "tenTusscherPanfilov",
        "imp_region[0].cellSurfVolRatio": 0.14,
        # Conductivities in S/m, monodomain from the harmonic mean of intra and extra
        "num_gregions": 1,
        "gregion[0].num_IDs": 1,
        "gregion[0].ID[0]": 1,
        "gregion[0].g_il": 0.17,
        "gregion[0].g_it": 0.019,
        "gregion[0].g_in": 0.019,
        "gregion[0].g_el": 0.62,
        "gregion[0].g_et": 0.24,
        "gregion[0].g_en": 0.24,
        # 50000 µA/cm^3 for 2 ms, as a transmembrane current: divided by 1400 /cm
        "num_stim": 1,
        "stim[0].crct.type": 0,
        "stim[0].elec.vtx_file": f"{base.name}_stimulus",
        "stim[0].ptcl.start": 0.0,
        "stim[0].ptcl.duration": 2.0,
        "stim[0].pulse.strength": 50000 / 1400,
        # Activation time per node: first upstroke through 0 mV
        "num_LATs": 1,
        "lats[0].ID": "activation",
        "lats[0].measurand": 0,
        "lats[0].all": 0,
        "lats[0].mode": 0,
        "lats[0].threshold": 0.0,
        "lats[0].method": 1,
    }


def check(activation: dolfinx.fem.Function):
    """Report and check the activation times on rank 0."""
    if (values := owned_values(activation)) is None:
        return
    x, t = values
    assert np.all(t >= 0), f"{np.count_nonzero(t < 0)} nodes did not activate"

    times = {}
    for name, point in POINTS.items():
        distance = np.linalg.norm(x - point, axis=1)
        times[name] = t[np.argmin(distance)]
        print(f"{name} {point}: {times[name]:6.2f} ms")
    assert times["P1"] < times["P9"] < times["P8"]
    assert 30 < times["P8"] < 90, "P8 activates far from the benchmark range"

    # Conduction velocity along the edges at y = z = 0 (along the fibers) and
    # x = z = 0 (across), away from the stimulus
    def velocity(axis, lo, hi):
        others = [i for i in range(3) if i != axis]
        edge = np.all(np.isclose(x[:, others], 0), axis=1) & (x[:, axis] > lo) & (x[:, axis] < hi)
        return np.polyfit(t[edge], x[edge, axis], 1)[0]  # mm/ms = m/s

    along, across = velocity(0, 5, 18), velocity(1, 3, 6.5)
    print(f"Conduction velocity {along:.3f} m/s along the fibers, {across:.3f} m/s across")
    assert along > 2 * across, "conduction is not faster along the fibers"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dx", type=float, default=0.2, help="Mesh resolution in mm")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "niederer")
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    if MPI.COMM_WORLD.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()
    base = outdir / "slab"

    write_mesh(base, args.dx)
    simulation = run_opencarp(outdir, parameters(base))
    *_, activation = read_mesh_and_activation(
        base, simulation / "init_acts_activation-thresh.dat", SCALE, outdir / "activation.xdmf"
    )
    check(activation)
    if MPI.COMM_WORLD.rank == 0:
        print("Niederer benchmark: OK")


if __name__ == "__main__":
    main()
