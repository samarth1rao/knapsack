#!/usr/bin/env python3
"""
Knapsack Algorithm Simulation and Analysis
Runs various knapsack algorithms and generates comprehensive visualisations
"""

import asyncio
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
import numpy as np
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
            "efficientalgo": {
                "executable": self.algorithms_path / "bin" / "efficientalgo",
                "name": "Efficient Algorithm",
                "sort_key": lambda n, w: n,  # n * np.log(n),
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
            "customtestbed": {
                "executable": self.algorithms_path / "bin" / "customtestbed",
                "name": "Custom Testbed",
                "sort_key": lambda n, w: n,
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
        self.memory_limit_gb = None

    def parse_list_string(self, s):
        """Parse string representation of list to actual list"""
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse list string: {s}, error: {e}")
            return []

    def _validate_inputs(self, algo_name, weights, values):
        """Validate algorithm inputs"""
        if not weights or not values:
            logger.error(f"Empty weights or values for {algo_name}")
            return False
        if len(weights) != len(values):
            logger.error(f"Weights and values length mismatch for {algo_name}")
            return False
        return True

    def _log_multiline_error(self, message, error_text):
        """Helper to log multi-line error messages"""
        logger.error(message)
        for line in error_text.split("\n"):
            if line.strip():
                logger.error(f"\t{line.strip()}")

    def _prepare_input_data(self, n, capacity, weights, values):
        """Prepare input data string for algorithm"""
        input_data = f"{n} {capacity}\n"
        input_data += " ".join(map(str, weights)) + "\n"
        input_data += " ".join(map(str, values)) + "\n"
        return input_data

    def _calculate_timeout(self, n, custom_timeout=None):
        """Calculate adaptive timeout based on problem size"""
        per_item = 0.5  # seconds per item (heuristic)
        timeout_seconds = min(
            120.0, max(2.0, self.base_timeout_seconds + per_item * float(n))
        )
        if custom_timeout is not None:
            timeout_seconds = float(custom_timeout)
        return timeout_seconds

    def _get_executable_path(self, algo_name):
        """Get the executable path for an algorithm, handling Windows .exe extension"""
        executable = self.algorithms[algo_name]["executable"]

        if platform.system() == "Windows" and not executable.exists():
            exe_path = executable.with_suffix(".exe")
            if exe_path.exists():
                executable = exe_path

        if not executable.exists():
            raise FileNotFoundError(f"Executable not found: {executable}")

        return executable

    def _build_command(self, executable):
        """Build command for execution, handling Windows+WSL compatibility"""
        if platform.system() == "Windows" and shutil.which("wsl"):
            path_str = str(executable).replace("\\", "/")
            m = re.match(r"^([A-Za-z]):(.*)$", path_str)
            if m:
                drive = m.group(1).lower()
                rest = m.group(2)
                wsl_path = f"/mnt/{drive}{rest}"
            else:
                wsl_path = path_str
            return ["wsl", wsl_path]
        return [str(executable)]

    def _get_resource_limiter(self, memory_divisor=1):
        """Get resource limiting function for Unix systems"""
        if platform.system() == "Windows" or self.memory_limit_gb is None:
            return None

        try:
            import resource

            def limit_resources():
                mem_bytes = int(
                    (self.memory_limit_gb * 1024 * 1024 * 1024) / memory_divisor  # pyright: ignore[reportOptionalOperand]
                )
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

            return limit_resources
        except ImportError:
            logger.warning("resource module not available, memory limiting disabled")
            return None

    def _parse_algorithm_output(self, output_str, algo_name, elapsed_us):
        """Parse algorithm output into structured result"""
        if not output_str:
            logger.warning(f"No output from {algo_name}")
            return None

        lines = output_str.strip().split("\n")

        try:
            max_value = int(lines[0].strip())
        except (ValueError, IndexError) as e:
            logger.error(f"Unexpected output format from {algo_name}: {e}")
            return None

        # Parse optional fields with defaults
        num_selected = 0
        selected_items = []
        execution_time = elapsed_us
        memory_used = 0

        if len(lines) > 1:
            try:
                num_selected = int(lines[1].strip())
            except (ValueError, IndexError):
                pass

        if num_selected > 0 and len(lines) > 2:
            try:
                selected_items = list(map(int, lines[2].split()))
            except (ValueError, IndexError):
                pass

        if len(lines) > 3:
            try:
                execution_time = int(lines[3])
            except (ValueError, IndexError):
                pass

        if len(lines) > 4:
            try:
                memory_used = int(lines[4])
            except (ValueError, IndexError):
                pass

        return {
            "max_value": max_value,
            "selected_items": selected_items,
            "execution_time": execution_time,
            "memory_used": memory_used,
            "success": True,
        }

    def load_dataset(self, dataset_name="knapsack_dataset.csv", category="Tiny"):
        """Load knapsack dataset and filter by category"""
        csv_path = self.data_path / dataset_name
        if not csv_path.exists():
            logger.error(f"Dataset file not found: {csv_path}")
            return None, 0

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            return None, 0

        total_rows = len(df)

        # Filter by category
        df_filtered = df[df["category"] == category].copy()

        if df_filtered.empty:
            logger.warning(f"No data found for category: {category}")
            return df_filtered, total_rows

        # Parse list columns with error handling
        for col in ["weights", "prices"]:
            if col in df_filtered.columns:
                df_filtered[col] = df_filtered[col].apply(self.parse_list_string)
            else:
                logger.error(f"Column '{col}' not found in dataset")
                return None, 0

        return df_filtered, total_rows

    def run_algorithm(
        self,
        algo_name,
        n,
        capacity,
        weights,
        values,
        custom_timeout=None,
        memory_divisor=1,
    ):
        """Run a specific algorithm with given inputs"""
        if algo_name not in self.algorithms:
            raise ValueError(f"Algorithm {algo_name} not found")

        # Validate inputs
        if not self._validate_inputs(algo_name, weights, values):
            return None

        # Get executable and build command
        executable = self._get_executable_path(algo_name)
        cmd = self._build_command(executable)

        # Prepare input and timeout
        input_data = self._prepare_input_data(n, capacity, weights, values)
        timeout_seconds = self._calculate_timeout(n, custom_timeout)

        # Get resource limiter if applicable
        preexec_fn = self._get_resource_limiter(memory_divisor)

        try:
            # Run and measure real elapsed time
            start = time.perf_counter()
            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=preexec_fn,
            )
            end = time.perf_counter()
            elapsed_us = int((end - start) * 1_000_000)

            if result.returncode != 0:
                self._log_multiline_error(f"Error running {algo_name}:", result.stderr)
                return None

            # Parse and return output
            return self._parse_algorithm_output(result.stdout, algo_name, elapsed_us)

        except subprocess.TimeoutExpired:
            logger.warning(f"Algorithm {algo_name} timed out (>{timeout_seconds:.1f}s)")
            return None
        except Exception as e:
            self._log_multiline_error(f"Error running {algo_name}:", str(e))
            return None

    async def run_algorithm_async(
        self,
        algo_name,
        n,
        capacity,
        weights,
        values,
        custom_timeout=None,
        memory_divisor=1,
    ):
        """Run a specific algorithm asynchronously with given inputs"""
        if algo_name not in self.algorithms:
            raise ValueError(f"Algorithm {algo_name} not found")

        # Validate inputs
        if not self._validate_inputs(algo_name, weights, values):
            return None

        # Get executable and build command
        executable = self._get_executable_path(algo_name)
        cmd = self._build_command(executable)

        # Prepare input and timeout
        input_data = self._prepare_input_data(n, capacity, weights, values)
        timeout_seconds = self._calculate_timeout(n, custom_timeout)

        # Get resource limiter if applicable
        preexec_fn = self._get_resource_limiter(memory_divisor)

        proc = None
        try:
            start = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=preexec_fn,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input_data.encode()), timeout=timeout_seconds
                )
                end = time.perf_counter()
                elapsed_us = int((end - start) * 1_000_000)

                if proc.returncode != 0:
                    self._log_multiline_error(
                        f"Error running {algo_name}:", stderr.decode()
                    )
                    return None

                # Parse and return output
                return self._parse_algorithm_output(
                    stdout.decode(), algo_name, elapsed_us
                )

            except asyncio.TimeoutError:
                await self._cleanup_process(proc)
                logger.warning(
                    f"Algorithm {algo_name} timed out (>{timeout_seconds:.1f}s)"
                )
                return None
        except Exception as e:
            await self._cleanup_process(proc)
            self._log_multiline_error(f"Error running {algo_name}:", str(e))
            return None
        finally:
            # Ensure process is properly cleaned up
            await self._cleanup_process(proc)

    async def _cleanup_process(self, proc):
        """Clean up an async subprocess"""
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

    def _initialize_results_dataframe(self, df):
        """Initialize results dataframe with all necessary columns"""
        results_df = df.copy()

        # Batch initialize numeric columns with numpy for better performance
        numeric_columns = []
        for algo_name in self.algorithms:
            numeric_columns.extend(
                [
                    f"{algo_name}_value",
                    f"{algo_name}_time",
                    f"{algo_name}_memory",
                    f"{algo_name}_accuracy",
                ]
            )

        # Initialize all numeric columns at once
        results_df[numeric_columns] = np.nan

        # Initialize object and boolean columns
        for algo_name in self.algorithms:
            results_df[f"{algo_name}_items"] = pd.Series(
                [None] * len(results_df), index=results_df.index, dtype=object
            )
            results_df[f"{algo_name}_optimal"] = False

        return results_df

    def _cleanup_event_loop(self, loop):
        """Properly cleanup an event loop"""
        try:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Wait for all tasks to complete with a timeout
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        except Exception:
            pass
        finally:
            loop.close()

    def simulate_all(
        self,
        dataset_name,
        category,
        max_parallel,
        custom_timeout=None,
        memory_limit_gb=None,
    ):
        """Run all algorithms on the dataset"""
        # Set memory limit for this run
        self.memory_limit_gb = memory_limit_gb

        logger.info(f"Loading {category} dataset from {dataset_name}...")
        df, _ = self.load_dataset(dataset_name, category)
        if df is None or df.empty:
            logger.error(f"No valid data for category {category}")
            return None

        logger.info(f"Found {len(df)} test cases.")

        # Set parallel tasks for this run
        self.max_parallel_tasks = max_parallel

        # Initialize results dataframe with all necessary columns
        results_df = self._initialize_results_dataframe(df)

        # Pre-calculate the run order for each algorithm
        run_orders = self._prepare_run_order(df)

        # Calculate memory divisor for parallel execution
        memory_divisor = max(1, max_parallel)  # Ensure at least 1
        if self.memory_limit_gb:
            logger.info(
                f"Parallel execution: {max_parallel} tasks, "
                f"memory limit per task: {self.memory_limit_gb / memory_divisor:.2f}GB"
            )

        # Run each algorithm independently
        for algo_name, sorted_indices in run_orders.items():
            algo_display_name = self.algorithms[algo_name]["name"]
            logger.info("=" * 60)
            logger.info(f"Running Algorithm: {algo_display_name}")
            logger.info("=" * 60)

            # Run with parallel execution
            try:
                # Create a new event loop for each algorithm run to avoid cleanup issues
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self._run_algorithm_parallel(
                            algo_name,
                            sorted_indices,
                            df,
                            results_df,
                            custom_timeout,
                            memory_divisor,
                            max_parallel,
                        )
                    )
                finally:
                    self._cleanup_event_loop(loop)
            except Exception as e:
                logger.error(f"Failed to run {algo_display_name}: {e}")
                continue

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
            plt.boxplot(accuracies, labels=algo_names, patch_artist=True)  # pyright: ignore[reportCallIssue]
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
        for line in summary_df.to_string(index=False).split("\n"):
            logger.info(line)

    def _prepare_run_order(self, df):
        """Pre-calculates the run order for all algorithms."""
        run_orders = {}
        # Work on a copy to avoid modifying original
        df_temp = df.copy()

        for algo_name, config in self.algorithms.items():
            complexity_col = f"{algo_name}_complexity"

            # Vectorised operation for better performance
            df_temp[complexity_col] = df_temp.apply(
                lambda row: config["sort_key"](row["n"], row["capacity"]), axis=1
            )

            # Store sorted indices
            run_orders[algo_name] = df_temp.sort_values(
                by=complexity_col, ascending=True
            ).index.tolist()

        return run_orders

    async def _run_algorithm_parallel(
        self,
        algo_name,
        sorted_indices,
        df,
        results_df,
        custom_timeout,
        memory_divisor,
        max_parallel,
    ):
        """Run algorithm tests in parallel with batching and failure tracking"""
        algo_display_name = self.algorithms[algo_name]["name"]
        semaphore = asyncio.Semaphore(max_parallel)

        total_tests = len(sorted_indices)
        failed_test_numbers = set()  # Track which test numbers have failed
        failures_lock = asyncio.Lock()
        test_count = 0
        should_stop = False

        def check_consecutive_failures(test_num):
            """Check if there are 3 consecutive test numbers that failed"""
            # Loop over all 3 windows
            for i in [-1, 0, 1]:
                # Generate the set of three consecutive test numbers and check if subset
                if {test_num + j + i for j in [-1, 0, 1]} <= failed_test_numbers:
                    # Return True if found
                    return True
            # No such window found
            return False

        async def run_single_test(idx, test_num):
            nonlocal should_stop, test_count

            # Check if we should stop before even starting
            if should_stop:
                return None

            async with semaphore:
                # Double-check after acquiring semaphore
                if should_stop:
                    return None

                row = df.loc[idx]
                logger.info(
                    f"{test_num}/{total_tests}: test_id={idx}, n={row['n']}, capacity={row['capacity']}"
                )

                result = await self.run_algorithm_async(
                    algo_name,
                    row["n"],
                    row["capacity"],
                    row["weights"],
                    row["prices"],
                    custom_timeout,
                    memory_divisor,
                )

                async with failures_lock:
                    if result:
                        if row["best_price"] == 0:
                            accuracy = 100.0 if result["max_value"] == 0 else 0.0
                        else:
                            accuracy = min(
                                100.0, (result["max_value"] / row["best_price"] * 100.0)
                            )

                        is_optimal = abs(result["max_value"] - row["best_price"]) < 1e-9

                        results_df.at[idx, f"{algo_name}_value"] = result["max_value"]
                        results_df.at[idx, f"{algo_name}_time"] = result[
                            "execution_time"
                        ]
                        results_df.at[idx, f"{algo_name}_memory"] = result[
                            "memory_used"
                        ]
                        results_df.at[idx, f"{algo_name}_items"] = result[
                            "selected_items"
                        ]
                        results_df.at[idx, f"{algo_name}_accuracy"] = accuracy
                        results_df.at[idx, f"{algo_name}_optimal"] = is_optimal

                        logger.info(
                            f"  -> {test_num}/{total_tests} Value: {result['max_value']}, "
                            f"Time: {result['execution_time']}μs, Accuracy: {accuracy:.2f}%, Optimal: {is_optimal}"
                        )
                        return True
                    else:
                        logger.warning(
                            f"  -> {test_num}/{total_tests} FAILED (test_id={idx})"
                        )
                        failed_test_numbers.add(test_num)

                        # Check for 3 consecutive failures
                        if check_consecutive_failures(test_num):
                            should_stop = True
                            logger.critical(
                                f"*** Algorithm {algo_display_name} hit 3 consecutive test failures "
                                f"(failed tests: {sorted(failed_test_numbers)}) - stopping ***"
                            )
                        return False

        # Launch all tasks at once - semaphore controls concurrency
        tasks = []
        for i, idx in enumerate(sorted_indices):
            if should_stop:
                break
            task = asyncio.create_task(run_single_test(idx, i + 1))
            tasks.append(task)

        # Process results as they complete
        try:
            for completed_task in asyncio.as_completed(tasks):
                try:
                    _ = await completed_task
                    test_count += 1

                    # If we should stop, cancel all remaining tasks
                    if should_stop:
                        cancelled_count = 0
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                                cancelled_count += 1

                        if cancelled_count > 0:
                            logger.info(f"Cancelled {cancelled_count} pending tasks")

                        # Wait for cancellations to complete with timeout
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(*tasks, return_exceptions=True),
                                timeout=5.0,
                            )
                        except asyncio.TimeoutError:
                            logger.warning("Some tasks did not cancel cleanly")

                        logger.info(
                            f"Skipping remaining {total_tests - test_count} test cases."
                        )
                        break

                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    logger.error(f"Exception in test: {e}")
                    # Exception counts as a failure - we don't know the test_num here
                    # but we can still set should_stop if needed
                    async with failures_lock:
                        # Since we don't have the test_num in this exception handler,
                        # we'll just log the error but not track it for consecutive failures
                        pass
        finally:
            # Ensure all tasks are complete
            await asyncio.gather(*tasks, return_exceptions=True)
            # Give a small delay to allow subprocess cleanup
            await asyncio.sleep(0.1)


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

    # --- DEFINE: memory limit (in GB, set to None to disable) ---
    memory_limit_gb = 48.0

    # --- DEFINE: simulation runs ---
    # Format: [dataset_name, category, max_parallel_tasks, timeout_seconds (optional)]
    # Note: memory limit will be split among parallel tasks
    simulation_runs = [
        # -- Default --
        ["knapsack_easy_dataset.csv", "ETiny", 1],
        # -- Easy --
        # ["knapsack_easy_dataset_l012_400.csv", "ETiny", 12, 14],
        # ["knapsack_easy_dataset_l012_400.csv", "ESmall", 12, 22],
        # ["knapsack_easy_dataset_l012_400.csv", "EMedium", 8, 30],
        # ["knapsack_easy_dataset_l3_40.csv", "ELarge", 8, 600],
        # -- Trap --
        # ["knapsack_trap_dataset_l012_400.csv", "TTiny", 12, 14],
        # ["knapsack_trap_dataset_l012_400.csv", "TSmall", 12, 22],
        # ["knapsack_trap_dataset_l012_400.csv", "TMedium", 8, 30],
        # ["knapsack_trap_dataset_l3_40.csv", "TLarge", 8, 600],
        # -- Hard1 --
        # ["knapsack_hard1_dataset.csv", "H1known", 8, 8],
        # ["knapsack_hard1_dataset.csv", "H1unknown", 8, 8],
        # -- Hard2 --
        # ["knapsack_hard2_dataset.csv", "H2xiang", 12, 14],
        # ["knapsack_hard2_dataset.csv", "H2pisingerlowdim", 12, 14],
        # ["knapsack_hard2_dataset.csv", "H2pisingerlarge", 8, 22],
    ]

    # Create simulator
    simulator = KnapsackSimulator(base_path)

    for run_config in simulation_runs:
        # Unpack config
        dataset_name = run_config[0]
        category = run_config[1]
        max_parallel = run_config[2]
        timeout = run_config[3] if len(run_config) > 3 else None

        logger.info(f"--- Running simulation for category: {category} ---")
        logger.info(f"Parallel tasks: {max_parallel}")
        if timeout:
            logger.info(f"Using timeout: {timeout}s")

        results_df = simulator.simulate_all(
            dataset_name=dataset_name,
            category=category,
            max_parallel=max_parallel,
            custom_timeout=timeout,
            memory_limit_gb=memory_limit_gb,
        )

        # Create visualisations only if we have results
        if results_df is not None and not results_df.empty:
            logger.info("Generating visualisations...")
            simulator.create_visualisations(results_df, category=category)
        else:
            logger.warning("No results to visualise for category %s (check dataset and runs).", category)

    logger.info("=" * 60)
    logger.info("All simulations completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
