#!/usr/bin/env python3
"""Launch raw Gazebo Classic gzclient without gazebo_ros GUI plugins."""

import os
import sys


def main() -> int:
    args = ["gzclient"]
    for arg in sys.argv[1:]:
        text = str(arg).strip().lower()
        if text in ("true", "1", "--verbose", "-v"):
            args.append("--verbose")
    os.execvp(args[0], args)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
