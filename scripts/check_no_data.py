"""CI guard: fail if documents, datasets or secrets are tracked by git.

Corpora are configuration only; their documents are downloaded at runtime. This keeps
licensed or confidential content out of the public history, even by accident.
"""

import subprocess
import sys

BLOCKED_SUFFIXES = (".pdf", ".docx", ".pptx", ".xlsx", ".jsonl", ".parquet", ".pkl", ".npy")
BLOCKED_NAMES = (".env",)
BLOCKED_DIRS = ("data/",)

files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
bad = [
    f
    for f in files.splitlines()
    if f.lower().endswith(BLOCKED_SUFFIXES)
    or f.split("/")[-1] in BLOCKED_NAMES
    or f.startswith(BLOCKED_DIRS)
]
if bad:
    print("Data or secret files are tracked by git:\n  " + "\n  ".join(bad))
    sys.exit(1)
print(f"OK: {len(files.splitlines())} tracked files, no data or secrets")
