#!/usr/bin/env python3
"""Launch the LRTIA GUI."""

import subprocess
import sys
from pathlib import Path

def main():
    gui_path = Path(__file__).parent.parent / "lrtia" / "gui.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(gui_path)])

if __name__ == "__main__":
    main()
