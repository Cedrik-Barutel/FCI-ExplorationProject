# Simulation Automation Toolkit (Dimensional & Nondimensional)

This repository provides a unified framework for running **dimensional** and **nondimensional** simulations, performing automated analysis, and generating result visualizations. Both workflows operate using plan CSV files and a shared `config.ini`, allowing flexible parameter sweeps and reproducible batch processing.

---

## ✨ Overview

The workflow consists of three main automated steps (for both dimensional and nondimensional modes):

1. **Run simulations** based on parameter combinations defined in plan CSV files.
2. **Run analysis** for completed simulations.
3. **Collect results** from subdirectories and generate summary heatmaps.

Both simulation modes follow the same structure and behave identically — they simply use different parameters (`diffusion/activity` vs. `Dam/Pe`) and different config sections.

---

# 🔧 Configuration Structure

All file paths and directories are configured through `config.ini`.

Two sections control the parameters:

```ini
[dim]
sim_plan_name = simulation_plan.csv
analyse_plan_name = analyse_plan.csv
directory = results/

[non_dim]
sim_plan_name = nondim_simulation_plan.csv
analyse_plan_name = nondim_analyse_plan.csv
directory = nondim_results/
```

### What each entry means

* **sim_plan_name** → which CSV contains the simulation parameters
* **analyse_plan_name** → which CSV contains the analysis parameters
* **directory** → where simulation outputs and final plots are stored

👉 If you change subdirectories or filenames, update them **only** in `config.ini`.

---

# 📁 Plan Files

Both dimensional and nondimensional modes use plan CSV files.

## 1. Dimensional Simulation Plan (`[dim] sim_plan_name`)

Expected columns:

* `diffusion`
* `activity`
* `done` *False*

## 2. Dimensional Analysis Plan (`[dim] analyse_plan_name`)

Expected columns:

* `diffusion`
* `k_off`
* `done` *False*

## 3. Nondimensional Simulation Plan (`[non_dim] sim_plan_name`)

Expected columns:

* `Dam`
* `Pe`
* `done` *False*

## 4. Nondimensional Analysis Plan (`[non_dim] analyse_plan_name`)

Expected columns:

* `Dam`
* `Pe`
* `done` *False*

### "done" column behavior

Rows marked `done=True` are skipped, allowing safe continuation after interruption.

---

# 🚀 Automation Functions

Both dimensional and nondimensional scripts follow the same architecture.

---

## 1. `auto_simulation()`

For each unfinished row in the simulation plan CSV:

* Reads parameter values
* Runs the corresponding simulation
* Saves results in the configured directory
* Marks row as done in the plan CSV
* Executes the matching analysis function
* Frees memory between runs

Used in:

* `dim`: `simulation(diff, act)`
* `non_dim`: `simulation(dam, pe)`

---

## 2. `auto_analysis()`

For each unfinished row in the analysis plan CSV:

* Reads the parameter pair
* Runs sweeping efficiency analysis (`run_sweeping_eff`)
* Saves updated plan CSV

Used in:

* `dim`: `run_sweeping_eff(diffusion, k_off)`
* `non_dim`: `run_sweeping_eff(Dam, Pe)`

---

## 3. `create_results()`

This function:

1. Reads all simulation subfolders in the configured `directory`
2. Extracts `results.csv` files
3. Merges them into a single DataFrame
4. Cleans or filters values
5. Generates two heatmaps:

   * **Sweeping efficiency**
   * **Derivative of fit**
6. Saves them to the output directory

Differences:

* **Dimensional results** use axes: `diffusion` × `k_off`
* **Nondimensional results** use axes: `Dam` × `Pe`

---

# ▶️ How to Use the Framework

## Step 1 — Configure `config.ini`

Decide whether you want to run dimensional or nondimensional simulations.
Update the corresponding section:

```ini
[non_dim]
sim_plan_name = nondim_simulation_plan.csv
analyse_plan_name = nondim_analyse_plan.csv
directory = nondim_results/
```

## Step 2 — Prepare your plan CSV files

Fill in the parameter combinations you want to simulate or analyse.

## Step 3 — Run the script

Depending on your project structure:

```bash
python main.py
```

```bash
python main_nondim.py
```

(or whichever executable file includes the two workflows)

## Step 4 — View results

Plots are saved to:

* `directory/sweeping_eff.png`
* `directory/derivative_fit.png`

Subdirectories contain individual simulation outputs.

---

# 📌 Notes & Features

* The script automatically resumes incomplete runs
* Matplotlib figures are closed between iterations to prevent memory leaks
* `gc.collect()` further reduces memory accumulation
* CSV plan files are rewritten after *every* completed row for robustness

---

# 🧩 Requirements

* Python 3.8+
* pandas
* numpy
* matplotlib
* configparser
* matplotlib
* dedalus
* lib_simulation
* scipy
* datetime
* csv
* os
* logging
* sys

Install them via:

```bash
pip install ***
```

