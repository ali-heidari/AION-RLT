# Assistant Interaction Convention

This convention defines how the agent should communicate during coding tasks.

1. The agent must state the overall task it is going to do before starting work.
2. The agent must explain each step it is doing in a clear, concise way.
3. Before writing code, the agent must describe the exact code change it plans to make.
4. After describing a code change, the agent must ask whether the developer understands or agrees before applying the change.
5. Once the developer confirms understanding, the agent applies the code and proceeds to the next step.

This workflow ensures transparent, interactive collaboration and prevents silent or unexpected code edits.

6. If the user says "sync instructions", the agent should reapply `instructions.md` from the standard repository to the current project, updating the local project instruction file(s) to match the canonical guidance.

7. The main agent instruction file for a project MUST be named `agent-instructions.md` and located in a folder named `.agent` at the project root. Agents working on the project MUST use and update ./.agent/agent-instructions.md as the master instruction. Any other instruction files for the agent must be placed inside the `.agent/` folder.