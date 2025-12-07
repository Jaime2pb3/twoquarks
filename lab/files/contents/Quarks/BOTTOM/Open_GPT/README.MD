# Open_GPT — TwoQuarks Orchestration Layer (Design Stage)

Status: design / not deployed

Purpose
Integrate all quark models (UP, DOWN, CHARM, STRANGE, TOP, BOTTOM)
into a unified GPT-style environment. This is the backend for the
"Open Lab" button.

Planned components
- Task router: maps tasks to quarks.
- Common API for all models.
- Experiment logger.
- Lightweight web UI for Open Lab.

Current state
- All individual quark models executed independently.
- Router API under design (see router.py).

Roadmap
1. Define standard interface: forward(), train(), eval().
2. Implement routing rules.
3. Build minimal HTTP API.
4. Attach UI to Open Lab button.

This is part of the future expansion of TwoQuarks.
