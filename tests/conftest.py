import os
import sys

# Ensure this repo's package directories are preferred when running tests.
# This helps pytest resolve imports like `simulation.*` to this copy of the repo
# when multiple copies with identical package names exist in the workspace.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Also add repo root's parent to sys.path for absolute imports if needed
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

print(f"[pytest conftest] added {REPO_ROOT} to sys.path for test import resolution")
