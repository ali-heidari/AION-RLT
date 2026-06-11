# Documentation Conventions

This repository treats documentation as a first-class artifact.

Key rules
- When code or standards change, update `README.md` and the relevant markdown files together.
- Use exact file paths in examples and references.
- Keep README content concise, structured, and easy to scan.
- Reference convention files from `README.md` so adopters can find detailed guidance.

Recommended structure
- `README.md` for adoption and high-level project purpose.
- `instructions.md` for agent-specific behavior guidance.
- Separate convention files for code, CI/CD, commits, and docs.

Documentation folder rule
- Every project adopting these conventions must include a `docs/` folder at the repository root.
- `docs/` must contain human-readable documentation files in Markdown.
- `docs/index.md` is required and must provide a list of links to the other files in the `docs/` folder.
