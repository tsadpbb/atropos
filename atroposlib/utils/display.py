"""Display utilities for formatting evaluation results and tables."""

from typing import Dict


def display_metrics_table(
    task_name: str, metrics: Dict, start_time: float, end_time: float
) -> None:
    """
    Display evaluation results in a formatted markdown table.

    Args:
        task_name: Name of the evaluation task
        metrics: Dictionary of metric names to values
        start_time: Start time of evaluation (unix timestamp)
        end_time: End time of evaluation (unix timestamp)
    """
    print("\n" + "=" * 84)
    print(f"Evaluation Results: {task_name}")
    print("=" * 84)

    # Column widths
    col_groups = 20
    col_version = 7
    col_filter = 6
    col_nshot = 6
    col_metric = 20
    col_dir = 3
    col_value = 10
    col_pm = 3
    col_stderr = 8

    # Header
    print(
        f"|{'Groups':<{col_groups}}|{'Version':<{col_version}}|{'Filter':<{col_filter}}|{'n-shot':<{col_nshot}}|{'Metric':<{col_metric}}|{'   ':<{col_dir}}|{'Value':>{col_value}}|{'   ':<{col_pm}}|{'Stderr':>{col_stderr}}|"  # noqa: E501
    )

    # Separator
    print(
        f"|{'-' * col_groups}|{'-' * col_version}|{'-' * col_filter}|{'-' * col_nshot}|{'-' * col_metric}|{'-' * col_dir}|{'-' * (col_value-1)}:|{'-' * col_pm}|{'-' * (col_stderr-1)}:|"  # noqa: E501
    )

    # Data rows
    for metric_name, metric_value in metrics.items():
        clean_metric_name = metric_name.replace("eval/", "").replace("_", " ")
        direction = "↑" if "correct" in metric_name or "acc" in metric_name else " "

        print(
            f"|{task_name:<{col_groups}}|{1:<{col_version}}|{'none':<{col_filter}}|{'':<{col_nshot}}|{clean_metric_name:<{col_metric}}|{direction:<{col_dir}}|{metric_value:>{col_value}.4f}|{'±':<{col_pm}}|{'0.0000':>{col_stderr}}|"  # noqa: E501
        )

    print("=" * 84)
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    print("=" * 84 + "\n")
