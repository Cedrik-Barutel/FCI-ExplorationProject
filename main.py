#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

This script runs the simulation first and then immediately
launches the analysis and visualization.
"""

import subprocess
import sys

def main():
    # 1) Run the simulation
    print("=== Starting simulation ===")
    subprocess.run([sys.executable, "lib_simulation.py"], check=True)

    # 2) Run analysis & visualization
    print("\n=== Starting analysis and visualization ===")
    subprocess.run([sys.executable, "generic_analyse_script.py"], check=True)

    print("\n=== All tasks completed ===")

if __name__ == "__main__":
    main()