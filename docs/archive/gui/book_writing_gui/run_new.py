import uvicorn
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import WitsV3 modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

if __name__ == "__main__":
    # Run the FastAPI server with a different module name
    uvicorn.run("main_new:app", host="0.0.0.0", port=8000, reload=True)
