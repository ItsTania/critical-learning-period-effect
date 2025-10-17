import pandas as pd
import numpy as np

def preprocess_experiment_df(df: pd.DataFrame, debug=False) -> pd.DataFrame:
    """
    Cleans the experiment DataFrame and adds epoch=0 rows for each run using initial metrics.
    
    Steps:
    1. Remove unwanted columns: 'Unnamed: 0', 'batches', 'train_batch_count', 'dur', 'train_loss_best'.
    2. For each run_name, initialisation, experiment_group_name, add a row for epoch=0 using initial metrics.
       - Maps 'initial_' columns to corresponding metric columns (removes '_logits').
    3. Remove the original 'initial_' columns after populating epoch=0.
    
    Args:
        df (pd.DataFrame): Original experiment DataFrame.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with epoch=0 rows added.
    """
    df_copy = df.copy()

    # Step 1: drop unwanted columns if they exist
    cols_to_drop = ['Unnamed: 0', 'batches', 'train_batch_count', 'dur', 'train_loss_best', 'run']
    df_copy = df_copy.drop(columns=[c for c in cols_to_drop if c in df_copy.columns])

    # Step 2: Identify initial metric columns
    initial_cols = [c for c in df_copy.columns if c.startswith("initial_")]

    # Map initial columns to corresponding metric columns
    initial_to_metric = {}
    for col in initial_cols:
        metric_name = col.replace("initial_", "").replace("_logits", "")
        if metric_name in df_copy.columns:
            initial_to_metric[metric_name] = col
    
    if debug:
        print("Initial -> metric map:", initial_to_metric)

    # Collect epoch=0 rows
    epoch0_rows = []

    # Group by run_name, initialisation, experiment_group_name
    group_cols = ["run_name", "initialisation", "experiment_group_name"]
    for _, group_df in df_copy.groupby(group_cols, sort=False):
        first_row = group_df.iloc[0]
        epoch0_row = first_row.to_dict()   # template
        epoch0_row["train_loss"] = None
        epoch0_row["epoch"] = 0

        # Fill in metrics from initial columns
        for metric, init_col in initial_to_metric.items():
            if init_col in group_df.columns:
                old_val = epoch0_row[metric]
                epoch0_row[metric] = first_row.get(init_col)
                if debug:
                    assert old_val != epoch0_row[metric]
                    print(f"group {first_row.get('run_name')}, set epoch0 {metric} = {epoch0_row[metric]} from {init_col}")

        epoch0_rows.append(epoch0_row)

    # Combine epoch=0 rows with original DataFrame
    df_epoch0 = pd.DataFrame(epoch0_rows)
    df_clean = pd.concat([df_epoch0, df_copy], ignore_index=True)

    # Step 3: drop the original 'initial_' columns
    df_clean = df_clean.drop(columns=initial_cols)

    # Sort by run_name, initialisation, experiment_group_name, epoch
    df_clean = df_clean.sort_values(by=group_cols + ["epoch"]).reset_index(drop=True)

    return df_clean


def get_summary_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean and standard error of metrics per epoch, grouped by 
    initialisation and experiment_group_name (averaging over run_name).

    Args:
        df (pd.DataFrame): Cleaned DataFrame with columns:
            ['epoch', 'metrics...', 'run_name', 'initialisation', 'experiment_group_name']

    Returns:
        pd.DataFrame: Summary DataFrame with columns:
            ['initialisation', 'experiment_group_name', 'epoch', 
             'metric1_mean', 'metric1_sem', 'metric2_mean', 'metric2_sem', ...]
    """
    # Metrics columns
    group_cols = ['initialisation', 'experiment_group_name', 'epoch']
    metric_cols = [c for c in df.columns if c not in group_cols + ['run_name']]

    # Aggregate: mean and standard error over run_name
    summary_list = []
    for (init, group_name, epoch), group in df.groupby(group_cols):
        summary_row = {
            'initialisation': init,
            'experiment_group_name': group_name,
            'epoch': epoch,
            'n_runs': group['run_name'].nunique()
        }
        for metric in metric_cols:
            summary_row[f"{metric}_mean"] = group[metric].mean()
            summary_row[f"{metric}_sem"] = group[metric].sem()  # standard error
        summary_list.append(summary_row)

    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values(by=['experiment_group_name', 'initialisation', 'epoch']).reset_index(drop=True)

    return summary_df, metric_cols


def smooth_runs(df: pd.DataFrame, smoothing_window=5, debug=False):
    """
    Smooths metric columns per run using moving average and returns a new df in same structure.

    Args:
        df (pd.DataFrame): Preprocessed experiment df (from preprocess_experiment_df)
        smoothing_window (int): Moving average window size
        debug (bool): Print debug information

    Returns:
        pd.DataFrame: Smoothed df with same structure and metadata
    """
    df_smooth = df.copy()

    # Identify metric columns to smooth (numeric metrics except epoch)
    metadata_cols = ["run_name", "initialisation", "experiment_group_name", "epoch"]
    metric_cols = [c for c in df.columns
                   if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])]

    if debug:
        print(f"Smoothing metrics: {metric_cols}")

    # Apply smoothing per run
    for (run_name, init, group), run_df_idx in df.groupby(["run_name", "initialisation", "experiment_group_name"]).groups.items():
        run_indices = list(run_df_idx)
        run_df = df.loc[run_indices].sort_values("epoch")

        if debug:
            print(f"Smoothing run={run_name}, init={init}, group={group}")

        for metric in metric_cols:
            series = run_df[metric]
            smoothed = series.rolling(window=smoothing_window, min_periods=1, center=True).mean()
            df_smooth.loc[run_indices, metric] = smoothed.values

    return df_smooth

def compute_mean_difference(df_summary: pd.DataFrame, metric: str, init1: str, init2: str):
    """
    Calculate the final epoch mean difference between two initialisations for each experiment group,
    along with the combined standard error.

    Args:
        df_summary (pd.DataFrame): Summary DataFrame with mean and sem columns.
        metric (str): Base metric name, e.g., "Achille_MNIST_test_target_acc".
        init1 (str): First initialisation (e.g., "random").
        init2 (str): Second initialisation (e.g., "source").
    
    Returns:
        pd.DataFrame: Columns = ['experiment_group_name', 'diff_mean', 'diff_sem']
    """
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"

    results = []
    
    groups = df_summary["experiment_group_name"].unique()
    for group in groups:
        df_group = df_summary[df_summary["experiment_group_name"] == group]
        

        df_init1 = df_group[df_group["initialisation"] == init1]
        df_init2 = df_group[df_group["initialisation"] == init2]
        max_epoch = df_init1["epoch"].max()

        assert max_epoch == df_init2["epoch"].max()

        if df_init1.empty or df_init2.empty:
            continue

        # Use the final epoch

        # Filter to only the max epoch row
        mean1 = df_init1[df_init1["epoch"] == max_epoch][mean_col].values[0]
        sem1  = df_init1[df_init1["epoch"] == max_epoch][sem_col].values[0]
        n1    = df_init1[df_init1["epoch"] == max_epoch]['n_runs'].values[0]

        mean2 = df_init2[df_init2["epoch"] == max_epoch][mean_col].values[0]
        sem2  = df_init2[df_init2["epoch"] == max_epoch][sem_col].values[0]
        n2    = df_init2[df_init2["epoch"] == max_epoch]['n_runs'].values[0]


        diff_mean = mean1 - mean2
        diff_sem  = np.sqrt(sem1**2 + sem2**2)  # combined standard error

        results.append({
            "experiment_group_name": group,
            "mean_init1": mean1,
            "n_init1": n1,
            "sem_init1": sem1,
            "mean_init2": mean2,
            "n_init2": n2,
            "sem_init2": sem2,
            "diff_mean": diff_mean,
            "diff_sem": diff_sem
        })

    return pd.DataFrame(results)

def print_diff_table(df_diff, show_percent=True):
    """
    Nicely prints the difference DataFrame in a tabular format.
    """
    # Header
    if show_percent: 
        header = (
            f"{'Experiment':<15} | "
            f"{'Init1 Mean':>10} | {'n':>3} | {'SE':>6} | "
            f"{'Init2 Mean':>10} | {'n':>3} | {'SE':>6} | "
            f"{'Performance Diff ± SE':>17}"
        )
    else:
        header = (
            f"{'Experiment':<15} | "
            f"{'Init1 Mean':>10} | {'n':>3} | {'SE':>6} | "
            f"{'Init2 Mean':>10} | {'n':>3} | {'SE':>6} | "
            f"{'Effect (%) ± SE':>17}"
        )

    print(header)
    print("-" * len(header))

    multiplier = 100 if show_percent else 1

    # Rows
    for _, row in df_diff.iterrows():
        effect_percent     = row['diff_mean'] * multiplier
        effect_sem_percent = row['diff_sem'] * multiplier
        print(
            f"{row['experiment_group_name']:<15} | "
            f"{row['mean_init1']:>10.4f} | {row['n_init1']:>3} | {row['sem_init1']:>6.4f} | "
            f"{row['mean_init2']:>10.4f} | {row['n_init2']:>3} | {row['sem_init2']:>6.4f} | "
            f"{effect_percent:>7.2f} ± {effect_sem_percent:.2f}"
        )

def diff_to_latex(df_diff, caption="Final epoch differences", label="tab:diff_table", show_percent=False):
    """
    Converts the df_diff DataFrame to a LaTeX table string.

    Args:
        df_diff (pd.DataFrame): Difference table with mean, SE, n, and effect.
        caption (str): Table caption
        label (str): Table label for referencing in LaTeX

    Returns:
        str: LaTeX table string
    """

    scale = 100 if show_percent else 1
    col_name = '\% Diff' if show_percent else 'Diff of Means'

    # Create a clean table for LaTeX
    latex_df = df_diff.copy()
    latex_df['Avg Performance Diff ± SE'] = (latex_df['diff_mean']*scale).round(2).astype(str) + " ± " + (latex_df['diff_sem']*scale).round(2).astype(str)
    
    # Select and rename columns
    latex_df = latex_df[[
        'experiment_group_name',
        'mean_init1', 'sem_init1', 'n_init1',
        'mean_init2', 'sem_init2', 'n_init2',
        'Avg Performance Diff ± SE'
    ]]

    cols = ['mean_init1', 'sem_init1', 'mean_init2', 'sem_init2']
    if show_percent:
        latex_df[cols] = latex_df[cols].apply(lambda x: x*100)

    latex_df.columns = [
        'Experiment', 'Init1 Mean', 'n', 'SE',
        'Init2 Mean', 'n', 'SE', f'{col_name} $\\pm$ SE'
    ]

    # Generate LaTeX string
    latex_str = latex_df.to_latex(index=False,
                                  caption=caption,
                                  label=label,
                                  float_format="%.3f",
                                  column_format='lrrrrrrr',
                                  escape=False)
    return latex_str
