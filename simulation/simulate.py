#!/usr/bin/env python3
"""
Knapsack Algorithm Simulation and Analysis
Runs various knapsack algorithms and generates comprehensive visualizations
"""

import ast
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better visualizations
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
        self.visualization_path = self.simulation_path / "visualizations"
        self.logs_path = self.simulation_path / "logs"

        # Create necessary directories
        self.results_path.mkdir(exist_ok=True)
        self.visualization_path.mkdir(exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)

        # Algorithm configurations
        # Add an entry for each executable placed in algorithms/bin
        self.algorithms = {
            "bruteforce": {
                "executable": self.algorithms_path / "bin" / "bruteforce",
                "name": "Brute Force",
            },
            "memoization": {
                "executable": self.algorithms_path / "bin" / "memoization",
                "name": "Memoization",
            },
            "dynamicprogramming": {
                "executable": self.algorithms_path / "bin" / "dynamicprogramming",
                "name": "Dynamic Programming",
            },
            "randompermutation": {
                "executable": self.algorithms_path / "bin" / "randompermutation",
                "name": "Random Permutation",
            },
            "meetinthemiddle": {
                "executable": self.algorithms_path / "bin" / "meetinthemiddle",
                "name": "Meet in the Middle",
            },
            "greedyheuristic": {
                "executable": self.algorithms_path / "bin" / "greedyheuristic",
                "name": "Greedy Heuristic",
            },
            "geneticalgorithm": {
                "executable": self.algorithms_path / "bin" / "geneticalgorithm",
                "name": "Genetic Algorithm",
            },
            # Add more algorithms as they are implemented
        }
        # Base timeout (seconds) used as part of adaptive timeout calculation
        self.base_timeout_seconds = 10.0

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
        df_filtered["best_picks"] = df_filtered["best_picks"].apply(
            self.parse_list_string
        )

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
            exe_path = executable.with_suffix('.exe')
            if exe_path.exists():
                executable = exe_path

        if not executable.exists():
            raise FileNotFoundError(f"Executable not found: {executable}")

        # Prepare input
        input_data = f"{n} {capacity}\n"
        input_data += " ".join(map(str, weights)) + "\n"
        input_data += " ".join(map(str, values)) + "\n"

        try:
            # Determine an adaptive timeout based on problem size n.
            # Heuristic: base + (per_item * n), capped to a reasonable maximum.
            per_item = 0.5  # seconds per item (heuristic)
            timeout_seconds = min(
                120.0, max(2.0, self.base_timeout_seconds + per_item * float(n))
            )
            if custom_timeout is not None:
                timeout_seconds = custom_timeout

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
        df, total_csv_rows = self.load_dataset(dataset_name, category)
        if df is None:
            return None

        # Sort instances by n
        df.sort_values(by="n", inplace=True)
        logger.info(f"Found {len(df)} test cases, sorted by problem size 'n'.")

        results = []
        consecutive_failures = {algo: 0 for algo in self.algorithms.keys()}
        excluded_algos = set()

        for counter, (idx, row) in enumerate(df.iterrows(), 1):
            n = row["n"]
            capacity = row["capacity"]
            weights = row["weights"]
            values = row["prices"]
            best_price = row["best_price"]
            seed = row["seed"]

            logger.info(
                f"Test case {idx + 1}/{total_csv_rows} ({counter}/{len(df)}): n={n}, capacity={capacity}"
            )

            test_result = {
                "test_id": idx,
                "n": n,
                "capacity": capacity,
                "optimal_value": best_price,
                "seed": seed,
            }

            for algo_name in self.algorithms.keys():
                if algo_name in excluded_algos:
                    logger.info(
                        f"  Skipping {self.algorithms[algo_name]['name']} (excluded after 3 consecutive failures)."
                    )
                    # Still need to add null results for this algo
                    test_result[f"{algo_name}_value"] = None
                    test_result[f"{algo_name}_time"] = None
                    test_result[f"{algo_name}_memory"] = None
                    test_result[f"{algo_name}_items"] = None
                    test_result[f"{algo_name}_accuracy"] = None
                    test_result[f"{algo_name}_optimal"] = False
                    continue

                logger.info(f"  Running {self.algorithms[algo_name]['name']}...")

                result = self.run_algorithm(
                    algo_name, n, capacity, weights, values, custom_timeout
                )

                if result:
                    consecutive_failures[algo_name] = 0  # Reset on success
                    test_result[f"{algo_name}_value"] = result["max_value"]
                    test_result[f"{algo_name}_time"] = result["execution_time"]
                    test_result[f"{algo_name}_memory"] = result["memory_used"]
                    test_result[f"{algo_name}_items"] = result["selected_items"]
                    test_result[f"{algo_name}_accuracy"] = (
                        100.0
                        if result["max_value"] == best_price
                        else (result["max_value"] / best_price * 100.0)
                    )
                    test_result[f"{algo_name}_optimal"] = (
                        result["max_value"] == best_price
                    )
                    logger.info(
                        f"  -> Value: {result['max_value']}, Time: {result['execution_time']}μs"
                    )
                else:
                    consecutive_failures[algo_name] += 1
                    logger.warning(
                        f"  -> FAILED ({consecutive_failures[algo_name]} consecutive)"
                    )
                    if consecutive_failures[algo_name] >= 3:
                        excluded_algos.add(algo_name)
                        logger.critical(
                            f"*** Algorithm {self.algorithms[algo_name]['name']} failed 3 times, excluding from further tests in this run. ***"
                        )

                    test_result[f"{algo_name}_value"] = None
                    test_result[f"{algo_name}_time"] = None
                    test_result[f"{algo_name}_memory"] = None
                    test_result[f"{algo_name}_items"] = None
                    test_result[f"{algo_name}_accuracy"] = None
                    test_result[f"{algo_name}_optimal"] = False

            results.append(test_result)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_path / f"results_{category}_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results for {category} saved to {results_file}")

        return results_df

    def create_visualizations(self, results_df, category="Tiny"):
        """Create comprehensive visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = self.visualization_path / f"{category}_{timestamp}"
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

        logger.info(f"Visualizations saved to {viz_dir}")

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
            plt.boxplot(accuracies, labels=algo_names, patch_artist=True)
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
                times_ms = df[time_col] / 1000.0
                accuracies = df[acc_col]

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
                rate = df[opt_col].sum() / len(df) * 100
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
            cellText=summary_df.values,
            colLabels=summary_df.columns,
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

        # Alternate row colors
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
        ["knapsack_test_tiny.csv", "Tiny"],
        # ["knapsack_dataset_l012_400.csv", "Tiny", 4],
        # ["knapsack_dataset_l012_400.csv", "Small", 8],
        # ["knapsack_dataset_l012_400.csv", "Medium", 15],
        # ["knapsack_dataset_l3_20.csv", "Large", 600],
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

        # Create visualizations
        logger.info("Generating visualizations...")
        simulator.create_visualizations(results_df, category=category)

    logger.info("=" * 60)
    logger.info("All simulations completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
