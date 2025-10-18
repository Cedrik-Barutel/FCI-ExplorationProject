import os
import pandas as pd
from dimensionsless import nondimensional as simulation
from generic_analyse_script import run_analysis as analyse  # call analysis without animations
from new_analyse_update_290925 import run_sweeping_eff
import gc
import matplotlib.pyplot as plt
import numpy as np

def auto_simulation():
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

def auto_analysis():
    cwd = os.getcwd()
    plan_path = f"{cwd}/analyse_plan.csv"
    plan = pd.read_csv(plan_path)
    # Ensure expected columns
    if "done" not in plan.columns:
        plan["done"] = False

    # Process sequentially instead of DataFrame.apply
    to_run_idx = plan.index[~plan["done"]]

    for idx in to_run_idx:
        row = plan.loc[idx]
        diff = row["diffusion"]
        k_off = row["k_off"]
        run_sweeping_eff(diff, k_off)
        # Mark as done and persist after each row
        plan.at[idx, "done"] = True
        plan.to_csv(plan_path, index=False)

        # Free memory between runs
        plt.close('all')
        gc.collect()


def create_results():
    cwd = os.getcwd()
    test_dir = f"{cwd}/test_nondim/"
    results_df = pd.DataFrame()
    for folder in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder)
        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if "results.csv" in file:
                df = pd.read_csv(os.path.join(folder_path, file))
                df = df.drop(df.columns[0], axis=1)
                results_df = pd.concat([results_df, df])
            else:
                continue
    # Drop rows with specific diffusion values
    results_df = results_df[~results_df['diffusion'].isin([0.03, 0.05, 0.1])]

    # Print to check the data
    print("Results DataFrame:")
    print(results_df.head())
    print("\nSweeping efficiency range:", results_df['sweeping efficiency'].min(), "-",
          results_df['sweeping efficiency'].max())
    print("Derivative fit range:", results_df['derivative_fit'].min(), "-", results_df['derivative_fit'].max())

    # Pivot the dataframe for sweeping efficiency
    pivot_df = results_df.pivot(index='k_off', columns='diffusion', values='sweeping efficiency')

    # Create the plot with square cells
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    im = ax.imshow(pivot_df.values, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im, label='Sweeping Efficiency')

    # Set tick labels to actual values
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)

    ax.set_xlabel('Diffusion')
    ax.set_ylabel('k_off')
    ax.set_title('Sweeping Efficiency')
    plt.tight_layout()
    plt.savefig(f"test_nondim/sweeping_eff.png")
    plt.show()

    # Pivot the dataframe for derivative_fit
    pivot_df_derivative = results_df.pivot(index='k_off', columns='diffusion', values='derivative_fit')

    # Create the plot for derivative_fit with potentially different colormap
    fig2, ax2 = plt.subplots(figsize=(8, 8), dpi=200)
    im2 = ax2.imshow(pivot_df_derivative.values, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im2, label='Derivative of Fit')

    # Set tick labels to actual values
    ax2.set_xticks(np.arange(len(pivot_df_derivative.columns)))
    ax2.set_xticklabels(pivot_df_derivative.columns)
    ax2.set_yticks(np.arange(len(pivot_df_derivative.index)))
    ax2.set_yticklabels(pivot_df_derivative.index)

    ax2.set_xlabel('Diffusion')
    ax2.set_ylabel('k_off')
    ax2.set_title('Derivative of Fit')
    plt.tight_layout()
    plt.savefig(f"test_nondim/derivative_fit.png")
    plt.show()
    return

if __name__ == "__main__":
    auto_simulation()
    auto_analysis()
    create_results()
