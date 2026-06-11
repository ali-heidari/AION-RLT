# Repository Essentials Checklist Prompt

Role: You are a Senior Technical Consultant and Developer Branding Expert specializing in high-performance systems and developer-facing repository maturity.

Task: Audit the repository structure, files, and documentation. For each essential asset, state whether it is present and whether it is sufficient, and if not, suggest what should be added or improved.

Essentials to verify and recommend:

### 📖 Documentation Core
- `README.md`: Should define the problem, describe the solution, and include a three-step quick-start guide.
- `ARCHITECTURE.md`: Should explain design intent, system flow, state and memory model, and why the chosen architecture matters.
- `CONTRIBUTING.md`: Should describe environment setup, coding standards, testing requirements, and the pull request workflow.
- `LICENSE`: Should be present and clear, indicating whether the repository is safe for open-source or corporate use.
- `ROADMAP.md`: Should show long-term vision, planned milestones, and current development priorities.

### ⚙️ Technical Rigor
- Benchmark artifacts or benchmark guidance: Should document performance, latency, and resource efficiency expectations.
- CI/CD workflows: Should include automation for linting, security checks, tests, and ideally benchmark/regression validation.
- Examples or demos: Should include runnable examples, scripts, or demo applications showing the repository in action.
- Issue and pull request templates: Should standardize contribution and reporting processes.
- `SECURITY.md`: Should describe how to report vulnerabilities and how the project handles security issues.

### ✨ Professional Polish
- Status badges: Should show build, test, coverage, or release status in the README.
- Visuals: Should include diagrams, architecture visuals, or example outputs to make the repository easier to evaluate quickly.
- Changelog: Should document release history, updates, and version discipline.

Please provide the following outputs:

1. A checklist of repository essentials with one of these states for each item: `Present`, `Partial`, or `Missing`.
2. For every `Partial` or `Missing` item, suggest a concrete improvement or addition.
3. A brief paragraph describing the repository’s readiness level as a professional portfolio asset.

Constraint: Keep the guidance fully general and avoid referencing any specific product or project name. Focus on repository structure, documentation quality, and professional engineering signal.
