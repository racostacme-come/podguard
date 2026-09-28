# Contributing

Create a focused feature branch from main. Use the configured Git author identity
and real commit dates. Keep synthetic examples independent of private/course data.

Before merging, run Ruff lint and format checks, pytest, the installed-wheel
example, the default CLI audit, `python -m build`, and `python -m pip check`.
Inspect changes to generated numerical results and figures. Include mathematical
justification and independent validation for changes to discretization or bounds.
Do not approve numerical changes on timing alone. Preserve meaningful branch
commits with a merge commit after checks pass; never invent review or test results.

Use an output directory such as `out` while developing. Regenerate the tracked
`results` intentionally, recording changed parameters and environment versions.
The default audit overwrites its named output files. Do not add handouts, course
solutions, credentials, virtual environments, or unrelated workspace files.
