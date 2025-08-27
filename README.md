# FCI-ExplorationProject

This project provides a numerical simulation of a filament–crosslinker model and offers analysis & visualization scripts to process the generated data.

## Table of Contents
1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Project Structure](#project-structure)  
4. [Usage](#usage)  
   - [1. Run Simulation](#1-run-simulation)  
   - [2. Analysis & Visualization](#2-analysis--visualization)  
   - [3. Combined Workflow](#3-combined-workflow)  
5. [Configuration](#configuration)

## Requirements
- Python 3.10  
- Virtual environment tool (e.g., `virtualenv`)  
- Required Python packages (see `requirements.txt` or `pyproject.toml`):  
  - dedalus  
  - numpy  
  - scipy  
  - pandas  
  - matplotlib  
  - pillow  
  - pyparsing  
  - six  

## Installation
1. Clone the repository  
   ```bash
   git clone <REPO-URL>
   cd FCI-ExplorationProject
   ```  
2. Create and activate a virtual environment  
   ```bash
   python3.10 -m virtualenv .venv
   source .venv/bin/activate
   ```  
3. Install dependencies
    
    3.1 Mac / Linux
   ```bash
   pip install -r requirements.txt
   ```  
    3.2 Windows
   ```bash
   pip install uv
   uv add -r requirements.txt
   ```
   
## Project Structure
- `generic_code.py`  
  Implements the filament–crosslinker simulation using the Dedalus PDE framework.  
- `generic_analyse_script.py`  
  Loads the resulting HDF5 files, computes key metrics, and generates plots & animations.  
- `lib_simulation.py`  
  Collection of helper functions (I/O, normalization, etc.) used by the other scripts.  
- `main.py`  
  Wrapper script that runs simulation and analysis back-to-back.  
- `requirements.txt`, `pyproject.toml`  
  Lists of required Python packages.  
- `.venv/`  
  Virtual environment directory.  

## Usage

### 1. Run Simulation
Generate an HDF5 output file with all relevant fields:

### 2. Analysis & Visualization
Load the HDF5 file, compute metrics (overlap, forces, velocities) and create figures (plots, heatmaps, GIFs):

### 3. Combined Workflow
Run both steps in sequence with a single command:


## Configuration
In `generic_analyse_script.py`, adjust the following paths if necessary:
- `dir_input_file`
- `name_input_file`
- `dir_output_file`

Ensure they match your directory and file naming conventions.
