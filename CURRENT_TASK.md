# Current Task: Initial Project Scaffolding

## Status: In Progress

## Goal
Set up the complete project structure, initialize git, create GitHub repository, and establish the development workflow files.

## Acceptance Criteria
- [x] Project directory structure created
- [x] pyproject.toml configured with all dependencies
- [x] __init__.py files created for all packages
- [x] Entry points created (pi/main.py, server/main.py)
- [x] Shared constants defined
- [x] .gitignore created
- [x] PROGRESS.md created
- [x] CURRENT_TASK.md created (this file)
- [ ] CONTEXT_RECOVERY.md created
- [ ] ARCHITECTURE.md created
- [ ] Git repository initialized
- [ ] GitHub repository created
- [ ] Initial commit pushed

## Files Created/Modified
- `pyproject.toml` - Poetry configuration with all dependencies
- `shared/constants.py` - Shared constants for timing, thresholds
- `pi/main.py` - Pi client entry point
- `server/main.py` - Server brain entry point
- `.gitignore` - Git ignore patterns
- `PROGRESS.md` - Development progress tracking
- `CURRENT_TASK.md` - Current task specification
- `docs/CONTEXT_RECOVERY.md` - Context recovery guide

## Implementation Notes
- Using Poetry for dependency management
- Separated dependencies into groups: core, server, pi, dev
- Pi-specific deps commented out (install on actual Pi hardware)

## Test Plan
- Verify Poetry can install dependencies: `poetry install`
- Verify project structure matches plan
