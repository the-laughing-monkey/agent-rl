#!/usr/bin/env python3
import sys
import disable_version_checks  # This monkey patches require_version

from swift.cli.rlhf import rlhf_main

if __name__ == "__main__":
    # Pass the command-line arguments to Swift's entry point
    rlhf_main()