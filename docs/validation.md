# Validation record

## Independent checks

The stencil test assembles neighbor differences directly from a padded 2D field
and compares against the Kronecker sparse matrix. Dense eigenvalues on three
small grids verify the analytic coercivity. The manufactured solution is
`sin(pi*x)*sin(2*pi*y)` with continuum load
`[pi^2*(kx+4*ky)+beta]*u`; deriving the load from the continuum, rather than the
assembled discrete matrix, makes this a truncation-error test.

For `kx=0.7, ky=1.6, beta=2`, grids 7, 15, 31 and 63 give successive L2 orders
near two. The finest order is 2.001767969. POD tests check weighted orthogonality,
optimal projection tail energy, rank rejection, zero load, linearity and full-basis
recovery. Random parameter tests include signed source weights and conductivities
outside training. QR residual norms are compared directly with `h*norm(f-A*u_r)`.

For the default study, 54 training snapshots yield numerical rank 54 using
`max(snapshot_shape)*eps*sigma_max`. Twenty retained modes discard a training
energy fraction of 2.2261e-10. The 36 held-out and 12 shifted samples are disjoint
from training; the fixed seed is 20260928. All L2, energy and mean bounds cover
their errors across five ranks (240 comparisons). The smallest L2 bound/error
ratio is 1.4763. The maximum QR/direct residual discrepancy is 2.78e-15.

Bounds are exact-arithmetic statements about the discrete system. Test comparisons
use an absolute tolerance of 1e-11 in the study (1e-12 in core random-case tests).
No assertion claims an interval-certified floating-point enclosure. Nodal source
sampling and continuum mesh error are outside the reduction bound. Galerkin energy
error decreases for nested bases; L2 error is not assumed monotone in general.

## Reproduction and test scope

The checked-in numerical CSV files are reproducible to floating-point tolerance
with the fixed parameters/seed. Timings and JSON package versions vary by system,
and PNG binaries can vary with rendering libraries. The image is visually checked
for legibility and includes separate continuum convergence and reduction bounds.

Timing includes fresh sparse assembly/factorization for each full-order query and
the reduced solve, QR bound and mean for each online query. Offline fitting and
field reconstruction are excluded from the online figure. Five batches of 48
queries follow one warmup per path. No timing threshold is asserted. A production
application with repeated identical coefficients could reuse factorizations or
use a tensor-product solver; this benchmark does not measure those alternatives.

On 2026-09-28, local Windows/Python 3.14.5 validation passed 51 tests in both the
development environment and a separate wheel environment; imports in the latter
were verified to resolve to site-packages. Statement coverage was 303/310 (97.74%)
for Python code, excluding compiled library internals and subprocess-only CLI lines.
Ruff lint/format, isolated source/wheel builds, dependency checks, the example and
the full CLI audit passed. Source-archive contents include the tests, CI and sample
results; the wheel contains only the package and metadata, with no environments.
The six-panel figure was visually inspected after shortening crowded titles.

CI runs Windows/Ubuntu and Python 3.11/3.14, checks formatting, tests development
and installed-wheel code, builds source/wheel archives, checks dependencies,
executes the API example and runs the full audit. Remote results are available in
the repository Actions tab after publication.

## Development history

The scaffold was committed on main. `feature/certified-rom` implemented the model,
POD and residual representation and merged after 46 tests and Ruff checks passed.
`feature/validation-study` adds the CLI, synthetic study, figures, packaging and
documentation. Initial new study formatting exceeded the configured line length;
Ruff formatting resolved it before checks. No external review is claimed.
