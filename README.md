# openCARP examples

Two checks that openCARP reads the files written by the io4dolfinx openCARP backend
and runs on them. Each script writes a mesh, region tags, fibers and a stimulus marker
with io4dolfinx, runs `openCARP` on them, reads the mesh and the activation times back
with io4dolfinx, writes `activation.xdmf` for ParaView, and checks the result.

| Script | Mesh | What the checks show was read |
| --- | --- | --- |
| `unit_square.py` | 1 cm square, triangles, stimulus on the left side | `.vtx` stimulus (planar wave), `.elem` regions (right half has a quarter of the conductivity, so the wave is about half as fast), `.lon` fibers |
| `niederer.py` | Niederer et al. (2011) benchmark slab, tetrahedra, stimulus in a 1.5 mm corner cube | vertex marker as `.vtx` stimulus, `.lon` fibers and sheets (faster along the fibers); prints the activation at P1-P9 |

```sh
python3 unit_square.py            # a few seconds
python3 niederer.py               # about 30 s at the default --dx 0.2 mm
mpirun -n 4 python3 niederer.py   # same files and results in parallel
```

You first need to make sure you have installed io4dolfinx and opencarp. For example with CONDA.

openCARP is taken from `$OPENCARP`, else from `PATH`, else from
`/opt/micromamba/envs/opencarp/bin/openCARP`. It runs serially, on rank 0. Output goes
to `unit_square/` and `niederer/` next to the scripts (`--outdir` to change). The
openCARP log is in `opencarp.log`, and its output in `simulation/`.

At `--dx 0.5` the Niederer slab fails the check: on that coarse mesh, conduction across
the fibers is too slow for the far side to activate within the 100 ms simulated.
