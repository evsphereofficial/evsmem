"""CLI entry point for evsmem."""
import os, sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import uvicorn
from main import app

def main():
    uvicorn.run(app, host="0.0.0.0", port=9876)

if __name__ == "__main__":
    main()
