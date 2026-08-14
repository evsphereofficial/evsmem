"""CLI entry point for evsmem."""
import os, sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import uvicorn
from main import app

def main():
    host = os.getenv("EVSMEM_HOST", "127.0.0.1")
    port = int(os.getenv("EVSMEM_PORT", "9876"))
    log_level = os.getenv("EVSMEM_LOG_LEVEL", "info")
    uvicorn.run(app, host=host, port=port, log_level=log_level)

if __name__ == "__main__":
    main()
