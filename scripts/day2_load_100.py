"""Day 2: load 100 English tickets from the public dataset's CSV into dataclasses,
print a count per queue, and save them as JSON.

The logic now lives in the package as the `capstone-ingest` command (Day 5);
this script stays so the Day 2 command keeps working.

    uv run python scripts/day2_load_100.py                       # download from Hugging Face
    uv run python scripts/day2_load_100.py --csv local_copy.csv  # or read a downloaded CSV
"""

import sys

from capstone_project.ingest import main

if __name__ == "__main__":
    sys.exit(main())
