"""
VS Code Debug Configuration
Copy this content to .vscode/launch.json to enable debugging
"""

DEBUG_CONFIG = {
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug: Agent Flow",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/debug_flow.py",
      "console": "integratedTerminal",
      "justMyCode": False,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Debug: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": False,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  ]
}

# To use:
# 1. Create .vscode/launch.json
# 2. Copy the DEBUG_CONFIG dictionary above (as JSON)
# 3. Open debug_flow.py in VS Code
# 4. Set breakpoints in agent files
# 5. Press F5 to start debugging
