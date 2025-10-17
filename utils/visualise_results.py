import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
import plotly.express as px
import math

def plot_two_metric_grid(df_summary, df_raw, metric_left, metric_right,
                         max_cols=1, alpha_sem=0.15, alpha_runs=0.05, base_colours=px.colors.qualitative.Dark24):
    """
    Creates a grid of subplots with two metrics: left column = metric_left, right column = metric_right.
    Each row corresponds to an experiment group. Initialization colors are fixed and one legend entry per type.
    
    Args:
        df_summary (pd.DataFrame): Summary dataframe with mean and SEM.
        df_raw (pd.DataFrame): Raw dataframe with individual runs.
        metric_left (str): Metric for the left column (e.g., accuracy).
        metric_right (str): Metric for the right column (e.g., loss).
        alpha_sem (float): Transparency for SEM shading.
        alpha_runs (float): Transparency for individual runs.
    """
    mean_cols = {metric_left: f"{metric_left}_mean", metric_right: f"{metric_right}_mean"}
    sem_cols = {metric_left: f"{metric_left}_sem", metric_right: f"{metric_right}_sem"}

    # Unique initialisations
    inits_unique = df_summary["initialisation"].unique()
    init_colours = {init: base_colours[i % len(base_colours)] for i, init in enumerate(inits_unique)}

    groups = df_summary["experiment_group_name"].unique()
    n_rows = len(groups)
    n_cols = 2  # left and right

    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[f"{g} - {metric_left}" if c==0 else f"{g} - {metric_right}" 
                                        for g in groups for c in range(2)],
                        shared_xaxes=False, shared_yaxes=False,
                        horizontal_spacing=0.08, vertical_spacing=0.12)

    # Track which init types have been added to legend
    legend_shown = {init: False for init in inits_unique}

    for row_idx, group_name in enumerate(groups):
        df_group_summary = df_summary[df_summary["experiment_group_name"] == group_name]
        df_group_raw = df_raw[df_raw["experiment_group_name"] == group_name]

        for col_idx, metric in enumerate([metric_left, metric_right]):
            mean_col = mean_cols[metric]
            sem_col = sem_cols[metric]
            col_num = col_idx + 1
            row_num = row_idx + 1

            for init in inits_unique:
                color = init_colours[init]

                # Individual runs
                df_init_raw = df_group_raw[df_group_raw["initialisation"] == init]
                for run_name, df_run in df_init_raw.groupby("run_name"):
                    rgba_run = mcolors.to_rgba(color, alpha=alpha_runs)
                    rgba_run_str = f'rgba({int(rgba_run[0]*255)}, {int(rgba_run[1]*255)}, {int(rgba_run[2]*255)}, {rgba_run[3]})'
                    fig.add_trace(go.Scatter(
                        x=df_run["epoch"],
                        y=df_run[metric],
                        mode='lines',
                        line=dict(color=rgba_run_str, width=1),
                        showlegend=False
                    ), row=row_num, col=col_num)

                # Mean line
                df_init_summary = df_group_summary[df_group_summary["initialisation"] == init]
                show_legend = not legend_shown[init] and col_idx == 0 and row_idx == 0
                fig.add_trace(go.Scatter(
                    x=df_init_summary["epoch"],
                    y=df_init_summary[mean_col],
                    mode='lines',
                    name=init if show_legend else None,
                    line=dict(color=color, width=2),
                    showlegend=show_legend
                ), row=row_num, col=col_num)
                legend_shown[init] = True

                # SEM shading
                y_upper = df_init_summary[mean_col] + df_init_summary[sem_col]
                y_lower = df_init_summary[mean_col] - df_init_summary[sem_col]
                rgba_sem = mcolors.to_rgba(color, alpha=alpha_sem)
                rgba_sem_str = f'rgba({int(rgba_sem[0]*255)}, {int(rgba_sem[1]*255)}, {int(rgba_sem[2]*255)}, {rgba_sem[3]})'
                fig.add_trace(go.Scatter(
                    x=list(df_init_summary["epoch"]) + list(df_init_summary["epoch"][::-1]),
                    y=list(y_upper) + list(y_lower[::-1]),
                    fill='toself',
                    fillcolor=rgba_sem_str,
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ), row=row_num, col=col_num)

            # Update axes
            fig.update_xaxes(title_text="Epoch", row=row_num, col=col_num)
            if col_idx == 0:
                fig.update_yaxes(title_text=metric, row=row_num, col=col_num)
            else:
                fig.update_yaxes(title_text="", row=row_num, col=col_num)

    fig.update_layout(
        title=f"{metric_left} vs {metric_right} across experiment groups",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Initialisation",
        height=300*n_rows  # adjust row height
    )

    return fig
