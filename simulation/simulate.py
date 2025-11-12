#!/usr/bin/env python3
"""
Knapsack Algorithm Simulation and Analysis
Runs various knapsack algorithms and generates comprehensive visualisations
"""

import ast
import logging
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style for better visualisations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


# Global logger
logger = logging.getLogger(__name__)


class KnapsackSimulator:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.algorithms_path = self.base_path / "algorithms"
        self.data_path = self.base_path / "data"
        self.simulation_path = self.base_path / "simulation"
        self.results_path = self.simulation_path / "results"
        self.visualisation_path = self.simulation_path / "visualisations"
        self.logs_path = self.simulation_path / "logs"

        # Create necessary directories
        self.results_path.mkdir(exist_ok=True)
        self.visualisation_path.mkdir(exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)

        # Algorithm configurations
        # Add an entry for each executable placed in algorithms/bin
        self.algorithms = {
            "bruteforce": {
                "executable": self.algorithms_path / "bin" / "bruteforce",
                "name": "Brute Force",
                "sort_key": lambda n, w: n,  # 2**n,
            },
            "memoization": {
                "executable": self.algorithms_path / "bin" / "memoization",
                "name": "Memoization",
                "sort_key": lambda n, w: n * w,
            },
            "dynamicprogramming": {
                "executable": self.algorithms_path / "bin" / "dynamicprogramming",
                "name": "Dynamic Programming",
                "sort_key": lambda n, w: n * w,
            },
            "branchandbound": {
                "executable": self.algorithms_path / "bin" / "branchandbound",
                "name": "Branch and Bound",
                "sort_key": lambda n, w: n,
            },
            "meetinthemiddle": {
                "executable": self.algorithms_path / "bin" / "meetinthemiddle",
                "name": "Meet in the Middle",
                "sort_key": lambda n, w: n,  # (2 ** (n / 2)) * n,
            },
            "greedyheuristic": {
                "executable": self.algorithms_path / "bin" / "greedyheuristic",
                "name": "Greedy Heuristic",
                "sort_key": lambda n, w: n,  # n * np.log(n),
            },
            "randompermutation": {
                "executable": self.algorithms_path / "bin" / "randompermutation",
                "name": "Random Permutation",
                "sort_key": lambda n, w: (n**1.5) * w,
            },
            "billionscale": {
                "executable": self.algorithms_path / "bin" / "billionscale",
                "name": "Billion Scale",
                "sort_key": lambda n, w: n,  # (n * M) + n * np.log(n),
            },
            "geneticalgorithm": {
                "executable": self.algorithms_path / "bin" / "geneticalgorithm",
                "name": "Genetic Algorithm",
                "sort_key": lambda n, w: n,  # n * P * G,
            },
            "customalgorithm": {
                "executable": self.algorithms_path / "bin" / "customalgorithm",
                "name": "Custom Algorithm",
                "sort_key": lambda n, w: n,
            },
        }
        # Base timeout (seconds) used as part of adaptive timeout calculation
        self.base_timeout_seconds = 10.0
        # Memory limit for subprocesses in GB. Set to None to disable.
        self.memory_limit_gb = 40

    def parse_list_string(self, s):
        """Parse string representation of list to actual list"""
        return ast.literal_eval(s)

    def load_dataset(self, dataset_name="knapsack_dataset.csv", category="Tiny"):
        """Load knapsack dataset and filter by category"""
        csv_path = self.data_path / dataset_name
        if not csv_path.exists():
            logger.error(f"Dataset file not found: {csv_path}")
            return None, 0

        df = pd.read_csv(csv_path)
        total_rows = len(df)

        # Filter by category
        df_filtered = df[df["category"] == category].copy()

        # Parse list columns
        df_filtered["weights"] = df_filtered["weights"].apply(self.parse_list_string)
        df_filtered["prices"] = df_filtered["prices"].apply(self.parse_list_string)

        return df_filtered, total_rows

    def run_algorithm(
        self, algo_name, n, capacity, weights, values, custom_timeout=None
    ):
        """Run a specific algorithm with given inputs"""
        if algo_name not in self.algorithms:
            raise ValueError(f"Algorithm {algo_name} not found")

        executable = self.algorithms[algo_name]["executable"]

        # Handle Windows .exe extension
        if platform.system() == "Windows" and not executable.exists():
            exe_path = executable.with_suffix(".exe")
            if exe_path.exists():
                executable = exe_path

        if not executable.exists():
            raise FileNotFoundError(f"Executable not found: {executable}")

        # Prepare input
        input_data = f"{n} {capacity}\n"
        input_data += " ".join(map(str, weights)) + "\n"
        input_data += " ".join(map(str, values)) + "\n"

        # Determine an adaptive timeout based on problem size n.
        # Heuristic: base + (per_item * n), capped to a reasonable maximum.
        per_item = 0.5  # seconds per item (heuristic)
        timeout_seconds = min(
            120.0, max(2.0, self.base_timeout_seconds + per_item * float(n))
        )
        # Custom: override if provided
        if custom_timeout is not None:
            timeout_seconds = custom_timeout

        try:
            # Resource limiting for the child process (Unix-only)
            if platform.system() != "Windows" and self.memory_limit_gb is not None:
                import resource

                def limit_resources():
                    # Set max virtual memory
                    mem_bytes = self.memory_limit_gb * 1024 * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            else:

                def limit_resources():
                    pass

            # Build the command. On Windows, prefer running via WSL when executable is a *nix binary.
            if platform.system() == "Windows" and shutil.which("wsl"):
                # Try to convert Windows path to a WSL-compatible path.
                path_str = str(executable).replace("\\", "/")
                m = re.match(r"^([A-Za-z]):(.*)$", path_str)
                if m:
                    drive = m.group(1).lower()
                    rest = m.group(2)
                    wsl_path = f"/mnt/{drive}{rest}"
                else:
                    wsl_path = path_str
                cmd = ["wsl", wsl_path]
            else:
                cmd = [str(executable)]

            # Run and measure real elapsed time as a fallback if the executable doesn't print time.
            start = time.perf_counter()
            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=limit_resources,
            )
            end = time.perf_counter()
            elapsed_us = int((end - start) * 1_000_000)

            if result.returncode != 0:
                # Print stderr for diagnostics and return failure for this run.
                logger.error(f"Error running {algo_name}: {result.stderr.strip()}")
                return None

            # Parse output safely. Expecting at least two lines (value and count), extras optional.
            out = result.stdout.strip()
            if not out:
                logger.warning(f"No output from {algo_name}")
                return None

            lines = out.split("\n")
            try:
                max_value = int(lines[0].strip())
            except Exception:
                logger.error(f"Unexpected output format from {algo_name}: {lines[:5]}")
                return None

            # selected count
            num_selected = 0
            selected_items = []
            if len(lines) > 1:
                try:
                    num_selected = int(lines[1].strip())
                except Exception:
                    # If second line isn't the count, ignore and try to parse items if present.
                    num_selected = 0

            if num_selected > 0 and len(lines) > 2:
                try:
                    selected_items = list(map(int, lines[2].split()))
                except Exception:
                    selected_items = []

            # Try to parse execution_time and memory if provided by the program; otherwise use measured
            execution_time = None
            memory_used = None
            if len(lines) > 3:
                try:
                    execution_time = int(lines[3])
                except Exception:
                    execution_time = None
            if len(lines) > 4:
                try:
                    memory_used = int(lines[4])
                except Exception:
                    memory_used = None

            if execution_time is None:
                execution_time = elapsed_us

            return {
                "max_value": max_value,
                "selected_items": selected_items,
                "execution_time": execution_time,  # microseconds
                "memory_used": memory_used or 0,  # bytes
                "success": True,
            }
        except subprocess.TimeoutExpired:
            logger.warning(f"Algorithm {algo_name} timed out (>{timeout_seconds:.1f}s)")
            return None
        except Exception as e:
            logger.error(f"Error running {algo_name}: {str(e)}")
            return None

    def simulate_all(self, dataset_name, category, custom_timeout=None):
        """Run all algorithms on the dataset"""
        logger.info(f"Loading {category} dataset from {dataset_name}...")
        df, _ = self.load_dataset(dataset_name, category)
        if df is None:
            return None

        logger.info(f"Found {len(df)} test cases.")

        # Start with the original dataframe and add result columns
        results_df = df.copy()
        for algo_name in self.algorithms:
            results_df[f"{algo_name}_value"] = pd.NA
            results_df[f"{algo_name}_time"] = pd.NA
            results_df[f"{algo_name}_memory"] = pd.NA
            results_df[f"{algo_name}_items"] = pd.Series(
                [None] * len(results_df), index=results_df.index, dtype=object
            )
            results_df[f"{algo_name}_accuracy"] = pd.NA
            results_df[f"{algo_name}_optimal"] = False

        # Pre-calculate the run order for each algorithm
        run_orders = self._prepare_run_order(df)

        # Run each algorithm independently
        for algo_name, sorted_indices in run_orders.items():
            algo_display_name = self.algorithms[algo_name]["name"]
            logger.info("=" * 60)
            logger.info(f"Running Algorithm: {algo_display_name}")
            logger.info("=" * 60)

            consecutive_failures = 0
            for test_count, idx in enumerate(sorted_indices, 1):
                if consecutive_failures >= 3:
                    logger.critical(
                        f"*** Algorithm {algo_display_name} discontinued after 3 consecutive failures. ***"
                    )
                    logger.info(
                        f"Skipping remaining {len(sorted_indices) - test_count + 1} test cases."
                    )
                    break

                row = df.loc[idx]
                logger.info(
                    f"Test {test_count}/{len(sorted_indices)}: test_id={idx}, n={row['n']}, capacity={row['capacity']}"
                )

                result = self.run_algorithm(
                    algo_name,
                    row["n"],
                    row["capacity"],
                    row["weights"],
                    row["prices"],
                    custom_timeout,
                )

                if result:
                    consecutive_failures = 0
                    accuracy = (
                        100.0
                        if row["best_price"] == 0 and result["max_value"] == 0
                        else (result["max_value"] / row["best_price"] * 100.0)
                        if row["best_price"] > 0
                        else 0.0
                    )
                    is_optimal = result["max_value"] == row["best_price"]

                    results_df.at[idx, f"{algo_name}_value"] = result["max_value"]
                    results_df.at[idx, f"{algo_name}_time"] = result["execution_time"]
                    results_df.at[idx, f"{algo_name}_memory"] = result["memory_used"]
                    results_df.at[idx, f"{algo_name}_items"] = result["selected_items"]
                    results_df.at[idx, f"{algo_name}_accuracy"] = accuracy
                    results_df.at[idx, f"{algo_name}_optimal"] = is_optimal

                    logger.info(
                        f"  -> Value: {result['max_value']}, Time: {result['execution_time']}μs, Accuracy: {accuracy:.2f}%, Optimal: {is_optimal}"
                    )
                else:
                    consecutive_failures += 1
                    logger.warning(
                        f"  -> FAILED (consecutive failures: {consecutive_failures})"
                    )
                    # Values are already NA/False, so no need to set them again

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_path / f"results_{category}_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results for {category} saved to {results_file}")

        return results_df

    def create_visualisations(self, results_df, category="Tiny"):
        """Create comprehensive visualisations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = self.visualisation_path / f"{category}_{timestamp}"
        viz_dir.mkdir(exist_ok=True)

        algo_list = list(self.algorithms.keys())

        # 1. Execution Time vs Problem Size
        self._plot_time_vs_size(results_df, algo_list, viz_dir)
        # 1b. Execution Time vs Knapsack Capacity
        self._plot_time_vs_capacity(results_df, algo_list, viz_dir)

        # 2. Solution Quality (Accuracy)
        self._plot_accuracy(results_df, algo_list, viz_dir)

        # 3. Memory Usage
        self._plot_memory(results_df, algo_list, viz_dir)

        # 4. Quality vs Time (Pareto Plot)
        self._plot_quality_vs_time(results_df, algo_list, viz_dir)

        # 5. Summary Statistics Table
        self._create_summary_table(results_df, algo_list, viz_dir)

        # 6. Optimality Rate
        self._plot_optimality_rate(results_df, algo_list, viz_dir)

        logger.info(f"Visualisations saved to {viz_dir}")

    def _plot_time_vs_size(self, df, algorithms, viz_dir):
        """Plot execution time vs problem size"""
        plt.figure(figsize=(12, 8))

        for algo in algorithms:
            time_col = f"{algo}_time"
            if time_col in df.columns:
                # Drop rows without time or n
                df_valid = df[["n", time_col]].dropna()
                if df_valid.empty:
                    continue
                # Convert microseconds to milliseconds
                times_ms = df_valid[time_col] / 1000.0
                plt.scatter(
                    df_valid["n"],
                    times_ms,
                    label=self.algorithms[algo]["name"],
                    s=60,
                    alpha=0.7,
                )

        plt.xlabel("Problem Size (N)", fontsize=12, fontweight="bold")
        plt.ylabel("Execution Time (ms)", fontsize=12, fontweight="bold")
        plt.title("Execution Time vs Problem Size", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(viz_dir / "time_vs_size.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_time_vs_capacity(self, df, algorithms, viz_dir):
        """Plot execution time vs knapsack capacity"""
        plt.figure(figsize=(12, 8))

        for algo in algorithms:
            time_col = f"{algo}_time"
            cap_col = "capacity"
            if time_col in df.columns and cap_col in df.columns:
                df_valid = df[[cap_col, time_col]].dropna()
                if df_valid.empty:
                    continue
                times_ms = df_valid[time_col] / 1000.0
                plt.scatter(
                    df_valid[cap_col],
                    times_ms,
                    label=self.algorithms[algo]["name"],
                    s=60,
                    alpha=0.7,
                )

        plt.xlabel("Knapsack Capacity", fontsize=12, fontweight="bold")
        plt.ylabel("Execution Time (ms)", fontsize=12, fontweight="bold")
        plt.title("Execution Time vs Knapsack Capacity", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(viz_dir / "time_vs_capacity.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_accuracy(self, df, algorithms, viz_dir):
        """Plot solution accuracy"""
        plt.figure(figsize=(12, 8))

        accuracies = []
        algo_names = []

        for algo in algorithms:
            acc_col = f"{algo}_accuracy"
            if acc_col in df.columns:
                acc_values = df[acc_col].dropna()
                if len(acc_values) > 0:
                    accuracies.append(acc_values.tolist())
                    algo_names.append(self.algorithms[algo]["name"])

        if accuracies:
            plt.boxplot(accuracies, tick_labels=algo_names, patch_artist=True)
            plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
            plt.title("Solution Quality Distribution", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3, axis="y")
            plt.ylim([95, 105])
            plt.axhline(y=100, color="r", linestyle="--", label="Optimal", linewidth=2)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                viz_dir / "accuracy_distribution.png", dpi=300, bbox_inches="tight"
            )
        plt.close()

    def _plot_memory(self, df, algorithms, viz_dir):
        """Plot memory usage"""
        plt.figure(figsize=(12, 8))

        for algo in algorithms:
            mem_col = f"{algo}_memory"
            if mem_col in df.columns:
                df_valid = df[["n", mem_col]].dropna()
                if df_valid.empty:
                    continue
                # Convert bytes to KB
                memory_kb = df_valid[mem_col] / 1024.0
                plt.scatter(
                    df_valid["n"],
                    memory_kb,
                    label=self.algorithms[algo]["name"],
                    s=50,
                    marker="s",
                    alpha=0.7,
                )

        plt.xlabel("Problem Size (N)", fontsize=12, fontweight="bold")
        plt.ylabel("Memory Usage (KB)", fontsize=12, fontweight="bold")
        plt.title("Memory Usage vs Problem Size", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / "memory_usage.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_quality_vs_time(self, df, algorithms, viz_dir):
        """Plot quality vs time (Pareto plot)"""
        plt.figure(figsize=(12, 8))

        for algo in algorithms:
            time_col = f"{algo}_time"
            acc_col = f"{algo}_accuracy"

            if time_col in df.columns and acc_col in df.columns:
                df_valid = df[[time_col, acc_col]].dropna()
                if df_valid.empty:
                    continue

                times_ms = df_valid[time_col] / 1000.0
                accuracies = df_valid[acc_col]

                plt.scatter(
                    times_ms,
                    accuracies,
                    label=self.algorithms[algo]["name"],
                    s=100,
                    alpha=0.6,
                    edgecolors="black",
                    linewidth=1.5,
                )

        plt.xlabel("Execution Time (ms)", fontsize=12, fontweight="bold")
        plt.ylabel("Solution Quality (%)", fontsize=12, fontweight="bold")
        plt.title(
            "Quality vs Time Trade-off (Pareto Plot)", fontsize=14, fontweight="bold"
        )
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(viz_dir / "quality_vs_time.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_optimality_rate(self, df, algorithms, viz_dir):
        """Plot optimality rate for each algorithm"""
        plt.figure(figsize=(10, 6))

        algo_names = []
        optimality_rates = []

        for algo in algorithms:
            opt_col = f"{algo}_optimal"
            if opt_col in df.columns:
                # Ensure there are non-NA values to avoid errors on empty dataframes
                valid_entries = df[opt_col].dropna()
                if not valid_entries.empty:
                    rate = valid_entries.sum() / len(valid_entries) * 100
                else:
                    rate = 0.0
                algo_names.append(self.algorithms[algo]["name"])
                optimality_rates.append(rate)

        if algo_names:
            bars = plt.bar(
                algo_names,
                optimality_rates,
                color="skyblue",
                edgecolor="black",
                linewidth=1.5,
            )
            plt.ylabel("Optimality Rate (%)", fontsize=12, fontweight="bold")
            plt.title(
                "Percentage of Optimal Solutions Found", fontsize=14, fontweight="bold"
            )
            plt.ylim([0, 105])
            plt.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar, rate in zip(bars, optimality_rates):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plt.tight_layout()
            plt.savefig(viz_dir / "optimality_rate.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_summary_table(self, df, algorithms, viz_dir):
        """Create summary statistics table"""
        summary_data = []

        for algo in algorithms:
            time_col = f"{algo}_time"
            acc_col = f"{algo}_accuracy"
            mem_col = f"{algo}_memory"
            opt_col = f"{algo}_optimal"

            if time_col in df.columns:
                algo_name = self.algorithms[algo]["name"]
                # Use dropna to avoid None/NaN interfering with stats
                times = pd.to_numeric(df[time_col], errors="coerce").dropna()
                avg_time_ms = (times.mean() / 1000.0) if not times.empty else 0
                max_time_ms = (times.max() / 1000.0) if not times.empty else 0
                min_time_ms = (times.min() / 1000.0) if not times.empty else 0
                avg_accuracy = (
                    pd.to_numeric(df[acc_col], errors="coerce").dropna().mean()
                    if acc_col in df.columns
                    else 0
                )
                mems = (
                    pd.to_numeric(df[mem_col], errors="coerce").dropna()
                    if mem_col in df.columns
                    else pd.Series(dtype=float)
                )
                avg_memory_kb = (mems.mean() / 1024.0) if not mems.empty else 0
                optimality_rate = (
                    (
                        pd.to_numeric(df[opt_col], errors="coerce").fillna(0).sum()
                        / len(df)
                        * 100
                    )
                    if opt_col in df.columns
                    else 0
                )

                summary_data.append(
                    {
                        "Algorithm": algo_name,
                        "Avg Time (ms)": f"{avg_time_ms:.3f}",
                        "Min Time (ms)": f"{min_time_ms:.3f}",
                        "Max Time (ms)": f"{max_time_ms:.3f}",
                        "Avg Accuracy (%)": f"{avg_accuracy:.2f}",
                        "Avg Memory (KB)": f"{avg_memory_kb:.2f}",
                        "Optimality Rate (%)": f"{optimality_rate:.1f}",
                    }
                )

        summary_df = pd.DataFrame(summary_data)

        # Save as CSV
        summary_df.to_csv(viz_dir / "summary_statistics.csv", index=False)

        # Create visual table
        fig, ax = plt.subplots(figsize=(14, len(summary_data) * 0.8 + 2))
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(
            cellText=summary_df.values.tolist(),
            colLabels=summary_df.columns.tolist(),
            cellLoc="center",
            loc="center",
            colWidths=[0.15] * len(summary_df.columns),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Alternate row colours
        for i in range(1, len(summary_data) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#f0f0f0")
                else:
                    table[(i, j)].set_facecolor("white")

        plt.title("Summary Statistics", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(viz_dir / "summary_table.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Summary Statistics:")
        logger.info(summary_df.to_string(index=False))

    def _prepare_run_order(self, df):
        """Pre-calculates the run order for all algorithms."""
        run_orders = {}
        for algo_name, config in self.algorithms.items():
            complexity_col = f"{algo_name}_complexity"

            df[complexity_col] = df.apply(
                lambda row: config["sort_key"](row["n"], row["capacity"]), axis=1
            )

            # Store sorted indices
            run_orders[algo_name] = df.sort_values(
                by=complexity_col, ascending=True
            ).index.tolist()
            # Drop the complexity column to keep the DataFrame clean
            df.drop(columns=[complexity_col], inplace=True)
        return run_orders


def main():
    # Get the base path (parent of simulation folder)
    base_path = Path(__file__).parent.parent

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = base_path / "simulation" / "logs"
    log_dir.mkdir(exist_ok=True)  # Ensure log directory exists
    log_file = log_dir / f"simulation_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger.info("=" * 60)
    logger.info("Knapsack Algorithm Simulation")
    logger.info("=" * 60)

    # --- Define the simulation runs here ---
    # Format: [dataset_name, category, optional_timeout_in_seconds]
    simulation_runs = [
        # -- Default --
        ["knapsack_dataset.csv", "Tiny"],
        # -- Easy --
        # ["knapsack_easy_dataset_l012_400.csv", "ETiny", 4],
        # ["knapsack_easy_dataset_l012_400.csv", "ESmall", 8],
        # ["knapsack_easy_dataset_l012_400.csv", "EMedium", 12],
        # ["knapsack_easy_dataset_l3_40.csv", "ELarge", 300],
        # # -- Trap --
        # ["knapsack_trap_dataset_l012_400.csv", "TTiny", 4],
        # ["knapsack_trap_dataset_l012_400.csv", "TSmall", 8],
        # ["knapsack_trap_dataset_l012_400.csv", "TMedium", 12],
        # ["knapsack_trap_dataset_l3_40.csv", "TLarge", 300],
        # -- Hard --
        # ["knapsack_hard_dataset.csv", "known", 1],
        # ["knapsack_hard_dataset.csv", "unknown", 8],
    ]

    # Create simulator
    simulator = KnapsackSimulator(base_path)

    for run_config in simulation_runs:
        # Unpack config
        dataset_name = run_config[0]
        category = run_config[1]
        timeout = run_config[2] if len(run_config) > 2 else None

        logger.info(f"--- Running simulation for category: {category} ---")
        if timeout:
            logger.info(f"Using hardcoded timeout: {timeout}s")

        results_df = simulator.simulate_all(
            dataset_name=dataset_name, category=category, custom_timeout=timeout
        )

        # Create visualisations
        logger.info("Generating visualisations...")
        simulator.create_visualisations(results_df, category=category)

    logger.info("=" * 60)
    logger.info("All simulations completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
