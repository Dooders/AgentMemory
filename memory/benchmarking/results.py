"""
Results handling for benchmarks.

This module provides functionality for storing, retrieving, and analyzing
benchmark results.
"""

import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class BenchmarkResults:
    """Class for managing benchmark results."""

    def __init__(self, results_dir: str = "benchmark_results"):
        """Initialize the results manager.

        Args:
            results_dir: Directory where results will be stored.
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_result(
        self,
        category: str,
        benchmark_name: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save benchmark results to file.

        Args:
            category: Category of the benchmark (e.g., 'storage', 'retrieval')
            benchmark_name: Name of the specific benchmark
            results: Dictionary containing the benchmark results
            metadata: Optional metadata about the benchmark run

        Returns:
            Path to the saved results file
        """
        # Create category directory if it doesn't exist
        category_dir = os.path.join(self.results_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        # Generate a timestamp for the results file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{benchmark_name}_{timestamp}.json"
        filepath = os.path.join(category_dir, filename)

        # Prepare the data to save
        data = {
            "benchmark": benchmark_name,
            "category": category,
            "timestamp": timestamp,
            "results": results,
            "metadata": metadata or {},
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_result(self, filepath: str) -> Dict[str, Any]:
        """Load benchmark results from file.

        Args:
            filepath: Path to the results file

        Returns:
            Dictionary containing the benchmark results
        """
        with open(filepath, "r") as f:
            return json.load(f)

    def list_results(
        self, category: Optional[str] = None, benchmark_name: Optional[str] = None
    ) -> List[str]:
        """List available result files.

        Args:
            category: Optional category filter
            benchmark_name: Optional benchmark name filter

        Returns:
            List of filepaths matching the criteria
        """
        if category:
            search_dir = os.path.join(self.results_dir, category)
            if not os.path.exists(search_dir):
                return []
        else:
            search_dir = self.results_dir

        results = []
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith(".json"):
                    if benchmark_name and not file.startswith(f"{benchmark_name}_"):
                        continue
                    results.append(os.path.join(root, file))

        return sorted(results)

    def compare_results(
        self, filepaths: List[str], metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare results from multiple benchmark runs.

        Args:
            filepaths: List of paths to result files to compare
            metrics: Optional list of metrics to compare

        Returns:
            DataFrame with comparison data
        """
        comparison_data = []

        for filepath in filepaths:
            data = self.load_result(filepath)

            # Extract filename without extension for labeling
            label = os.path.basename(filepath).split(".")[0]

            result_data = {
                "run": label,
                "benchmark": data.get("benchmark", "unknown"),
                "category": data.get("category", "unknown"),
                "timestamp": data.get("timestamp", "unknown"),
            }

            # Flatten results structure for easier comparison
            results = data.get("results", {})
            flat_results = self._flatten_dict(results, prefix="")

            # Filter metrics if specified
            if metrics:
                flat_results = {k: v for k, v in flat_results.items() if k in metrics}

            result_data.update(flat_results)
            comparison_data.append(result_data)

        return pd.DataFrame(comparison_data)

    def generate_report(
        self,
        category: Optional[str] = None,
        benchmark_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Generate an HTML report for benchmark results.

        Args:
            category: Optional category filter
            benchmark_name: Optional benchmark name filter
            output_dir: Directory to save the report (defaults to results_dir/reports)

        Returns:
            Path to the generated report
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)

        # Get matching result files
        result_files = self.list_results(category, benchmark_name)
        if not result_files:
            raise ValueError(
                f"No results found for category={category}, benchmark={benchmark_name}"
            )

        # Load all results
        results_data = [self.load_result(filepath) for filepath in result_files]

        # Generate timestamp for the report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine report filename
        if category and benchmark_name:
            report_name = f"{category}_{benchmark_name}_{timestamp}.html"
        elif category:
            report_name = f"{category}_{timestamp}.html"
        elif benchmark_name:
            report_name = f"{benchmark_name}_{timestamp}.html"
        else:
            report_name = f"benchmark_report_{timestamp}.html"

        report_path = os.path.join(output_dir, report_name)

        # Generate the report
        df = self.compare_results(result_files)
        report_html = self._generate_html_report(df, category, benchmark_name)

        with open(report_path, "w") as f:
            f.write(report_html)

        return report_path

    def plot_comparison(
        self,
        filepaths: List[str],
        metric: str,
        x_axis: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot a comparison of a specific metric across benchmark runs.

        Args:
            filepaths: List of paths to result files to compare
            metric: The metric to plot
            x_axis: Optional x-axis parameter to group by
            title: Optional plot title
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib figure object
        """
        df = self.compare_results(filepaths)

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        fig, ax = plt.subplots(figsize=figsize)

        if x_axis and x_axis in df.columns:
            # Plot metric vs x_axis for each run
            for run in df["run"].unique():
                run_data = df[df["run"] == run]
                ax.plot(run_data[x_axis], run_data[metric], marker="o", label=run)
            ax.set_xlabel(x_axis)
        else:
            # Plot metric for each run as bar chart
            ax.bar(df["run"], df[metric])
            ax.set_xlabel("Benchmark Run")

        ax.set_ylabel(metric)
        ax.set_title(title or f"Comparison of {metric}")

        if x_axis:
            ax.legend()

        plt.tight_layout()
        return fig

    def save_plot(
        self, fig: plt.Figure, filename: str, output_dir: Optional[str] = None
    ) -> str:
        """Save a plot to file.

        Args:
            fig: Matplotlib figure object
            filename: Filename for the plot
            output_dir: Directory to save the plot (defaults to results_dir/plots)

        Returns:
            Path to the saved plot
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)

        if not filename.endswith((".png", ".jpg", ".svg", ".pdf")):
            filename += ".png"

        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        return filepath

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten a nested dictionary structure.

        Args:
            d: Dictionary to flatten
            prefix: Prefix for flattened keys

        Returns:
            Flattened dictionary
        """
        result = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k

            if isinstance(v, dict):
                result.update(self._flatten_dict(v, f"{key}_"))
            elif isinstance(v, (list, tuple)) and all(
                isinstance(x, (int, float)) for x in v
            ):
                # For lists of numbers, store the average and max
                result[f"{key}_avg"] = sum(v) / len(v) if v else 0
                result[f"{key}_max"] = max(v) if v else 0
                result[f"{key}_min"] = min(v) if v else 0
            else:
                result[key] = v

        return result

    def _generate_html_report(
        self, df: pd.DataFrame, category: Optional[str], benchmark_name: Optional[str]
    ) -> str:
        """Generate HTML report content.

        Args:
            df: DataFrame with benchmark comparison data
            category: Category filter used
            benchmark_name: Benchmark name filter used

        Returns:
            HTML content for the report
        """
        title = "Benchmark Report"
        if category and benchmark_name:
            title += f" for {category} - {benchmark_name}"
        elif category:
            title += f" for {category}"
        elif benchmark_name:
            title += f" for {benchmark_name}"

        # Basic HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <div class="summary">
                    <p>Report generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>Number of benchmark runs: {len(df['run'].unique())}</p>
                </div>
                
                <h2>Comparison Table</h2>
                {df.to_html(index=False)}
            </div>
        </body>
        </html>
        """

        return html
