#!/usr/bin/env python3
import os
import glob

DIRS = ["logs", "reports"]

for d in DIRS:
    files = glob.glob(os.path.join(d, "*"))
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
            print(f"removed: {f}")

print("done.")
