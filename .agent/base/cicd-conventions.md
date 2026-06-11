# CI/CD Conventions

This repository does not include CI by default. If you add CI examples, follow these conventions.

Placement
- Put workflow files under `.github/workflows/`.
- Keep workflow docs and references in `README.md`.

General rules
- Do not invent CI steps unless the repository contains explicit build or test files.
- Keep CI examples simple and specific to the supported languages or tools.
- Reference any workflow files from `README.md` so adopters know where to find them.
