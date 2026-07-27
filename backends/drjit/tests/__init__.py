"""Dr.Jit backend test package.

`backends/drjit/tests` and the repository-root `tests` are two directories that
both claim the top-level package name `tests`. Which one wins depends on where
the interpreter was started: the repository-root invocation documented in
`README.md` (`python -m unittest backends.drjit.tests.drjit.<module>`) binds
`tests` to the root directory, while the backend-local invocations
(`python -m unittest tests.drjit.<module>` as documented in
`backends/drjit/README.md`, and the CI-style
`python -m unittest discover -s tests -t .` run from `backends/drjit`) bind it
here. Under the
backend-local invocation the shared golden fixtures at the repository root were
unreachable, so `tests.drjit.test_golden_scenes`, `tests.drjit.test_cuda_trace_backend`,
and `tests.drjit.test_trace_backend_gate` failed to import and never ran.

Extending this package's search path with the repository-root `tests` directory
makes the two halves resolve as a single package, so `tests.golden` names the
same modules regardless of the starting directory. The root directory is
appended rather than prepended, so this package's own modules always win, and
`unittest discover` still walks only the directory it was pointed at.
"""

from pathlib import Path as _Path

__path__.append(str(_Path(__file__).resolve().parents[3] / "tests"))
