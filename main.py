import os
import pandas as pd
from new_code import run_simulation as simulation
from generic_analyse_script import run_analysis as analyse  # call analysis without animations
import gc
import matplotlib.pyplot as plt

def main():
    cwd = os.getcwd()
    plan_path = f"{cwd}/simulation_plan.csv"
    plan = pd.read_csv(plan_path)

    # Ensure expected columns
    if "done" not in plan.columns:
        plan["done"] = False

    # Process sequentially instead of DataFrame.apply
    to_run_idx = plan.index[~plan["done"]]

    for idx in to_run_idx:
        row = plan.loc[idx]
        diff = row["diffusion"]
        act = row["activity"]

        # Run simulation
        h5_path = simulation(diff, act)

        try:
            analyse(diff, act, bool_anim=True)
        except TypeError:
            # Fallback if signature differs
            analyse(diff, act)

        # Mark as done and persist after each row
        plan.at[idx, "done"] = True
        plan.to_csv(plan_path, index=False)

        # Free memory between runs
        plt.close('all')
        gc.collect()


if __name__ == "__main__":
    main()
