# Conventions

- Existing modules use plain functions, dictionaries, pandas DataFrames, and module-level configuration.
- Imports often assume execution from `py/` or insert the `py` directory into `sys.path`.
- Existing user-facing documentation is mostly Portuguese; new specification and implementation notes use concise English, while domain terminology remains aligned with `CONTEXT.md`.
- Outputs are written under `outputs/`; cached/intermediate data lives under `.cache/` or `data/`.
- Existing code is not uniformly formatted or test-driven, so new code should keep its seam small and avoid broad refactors.

