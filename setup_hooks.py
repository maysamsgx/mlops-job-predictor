import os
import stat

HOOK_PATH = os.path.join('.git', 'hooks', 'pre-commit')

HOOK_CONTENT = """#!/bin/sh
echo "Running pre-commit hook: Fixing Lint..."
python fix_lint.py

echo "Checking Code Quality..."
python -m pylint src inference_service.py train_pipeline.py --fail-under=7.0 --ignore=.venv
"""

def install_hook():
    if not os.path.isdir('.git'):
        print("Error: .git directory not found. Are you in the project root?")
        return

    with open(HOOK_PATH, 'w') as f:
        f.write(HOOK_CONTENT)
    
    # Make executable (Windows doesn't use chmod the same way but good for cross-compat logic mindset)
    # On Windows, git bash uses the shebang.
    print(f"Hook installed at {HOOK_PATH}")

if __name__ == "__main__":
    install_hook()
