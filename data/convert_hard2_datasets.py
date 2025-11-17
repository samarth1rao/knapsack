#!/usr/bin/env python3
"""
Script to process knapsack problem instances and generate CSV datasets.
Processes all testcases from data/gh_knapsack-01-instances directory.
"""

import csv
from pathlib import Path


def parse_instance_file(instance_path, has_solution=False):
    """Parse instance file and extract n, weights, prices, capacity, and optional solution.
    Format: First line is 'n capacity', followed by n lines of 'price weight'.
    Large-scale files have an additional line with binary solution (space-separated 0s and 1s).
    """
    with open(instance_path, "r") as f:
        lines = f.readlines()

    # First line: n capacity
    first_line = lines[0].strip().split()
    n = int(first_line[0])
    capacity = int(first_line[1])

    weights = []
    prices = []

    # Next n lines: price weight
    for i in range(1, n + 1):
        parts = lines[i].strip().split()
        try:
            # Handle both integer and float values (convert floats to ints)
            price = int(float(parts[0]))
            weight = int(float(parts[1]))
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Error parsing line {i + 1} in {instance_path}: {lines[i].strip()}"
            ) from e
        prices.append(price)
        weights.append(weight)

    # Check if there's a solution line (binary string)
    best_picks = []
    if has_solution and len(lines) > n + 1:
        solution_line = lines[n + 1].strip()
        if solution_line:  # Not empty
            binary_solution = solution_line.split()
            # Convert to list of indices where value is 1
            best_picks = [i for i, val in enumerate(binary_solution) if val == "1"]

    return n, weights, prices, capacity, best_picks


def parse_optimum_file(optimum_path):
    """Parse optimum file and extract the best price value.
    Returns the optimal value if file exists, otherwise None.
    """
    try:
        with open(optimum_path, "r") as f:
            return int(f.read().strip())
    except (ValueError, FileNotFoundError):
        return None


def main():
    # Setup paths
    base_dir = Path(__file__).parent
    instances_dir = base_dir / "gh_knapsack-01-instances"
    output_file = base_dir / "knapsack_hard2_dataset.csv"

    # List to store all data
    all_data = []

    # Process Pisinger instances
    print("Processing Pisinger instances...")
    pisinger_dir = instances_dir / "pisinger_instances_01_KP"

    # Process large_scale instances
    large_scale_dir = pisinger_dir / "large_scale"
    large_scale_optimum_dir = pisinger_dir / "large_scale-optimum"

    if large_scale_dir.exists():
        instance_files = sorted([f for f in large_scale_dir.iterdir() if f.is_file()])
        for idx, instance_file in enumerate(instance_files, 1):
            if idx % 10 == 0:
                print(
                    f"  Processed {idx}/{len(instance_files)} large-scale instances..."
                )

            # Parse instance (large-scale files have binary solution)
            n, weights, prices, capacity, best_picks = parse_instance_file(
                instance_file, has_solution=True
            )

            # Check for optimum
            optimum_file = large_scale_optimum_dir / instance_file.name
            best_price = parse_optimum_file(optimum_file)

            all_data.append(
                {
                    "category": "H2pisingerlarge",
                    "n": n,
                    "weights": str(weights),
                    "prices": str(prices),
                    "capacity": capacity,
                    "best_picks": str(best_picks) if best_picks else "",
                    "best_price": best_price if best_price is not None else "",
                }
            )

    # Process low-dimensional instances
    low_dim_dir = pisinger_dir / "low-dimensional"
    low_dim_optimum_dir = pisinger_dir / "low-dimensional-optimum"

    if low_dim_dir.exists():
        instance_files = sorted([f for f in low_dim_dir.iterdir() if f.is_file()])
        for idx, instance_file in enumerate(instance_files, 1):
            # Parse instance (low-dimensional files don't have solution)
            n, weights, prices, capacity, best_picks = parse_instance_file(
                instance_file, has_solution=False
            )

            # Check for optimum
            optimum_file = low_dim_optimum_dir / instance_file.name
            best_price = parse_optimum_file(optimum_file)

            all_data.append(
                {
                    "category": "H2pisingerlowdim",
                    "n": n,
                    "weights": str(weights),
                    "prices": str(prices),
                    "capacity": capacity,
                    "best_picks": "",  # Solution not available for low-dimensional
                    "best_price": best_price if best_price is not None else "",
                }
            )

    large_scale_count = len([d for d in all_data if d["category"] == "H2pisingerlarge"])
    low_dim_count = len([d for d in all_data if d["category"] == "H2pisingerlowdim"])
    print(
        f"Processed {large_scale_count} large-scale + {low_dim_count} low-dimensional Pisinger instances"
    )

    # Process Xiang instances
    print("\nProcessing Xiang instances...")
    xiang_dir = instances_dir / "xiang_instances_01_KP"

    if xiang_dir.exists():
        instance_files = sorted([f for f in xiang_dir.iterdir() if f.is_file()])
        for idx, instance_file in enumerate(instance_files, 1):
            # Parse instance (Xiang files don't have solution)
            n, weights, prices, capacity, best_picks = parse_instance_file(
                instance_file, has_solution=False
            )

            all_data.append(
                {
                    "category": "H2xiang",
                    "n": n,
                    "weights": str(weights),
                    "prices": str(prices),
                    "capacity": capacity,
                    "best_picks": "",  # Solution not available
                    "best_price": "",  # Optimum not available for Xiang instances
                }
            )

    print(
        f"Processed {len([d for d in all_data if d['category'] == 'H2xiang'])} Xiang instances"
    )

    # Sort by category and n
    all_data.sort(key=lambda x: (x["category"], x["n"]))

    # Write all data to CSV
    print(f"\nWriting {output_file}...")
    with open(output_file, "w", newline="") as f:
        fieldnames = [
            "category",
            "n",
            "weights",
            "prices",
            "capacity",
            "best_picks",
            "best_price",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    large_scale_count = len([d for d in all_data if d["category"] == "H2pisingerlarge"])
    low_dim_count = len([d for d in all_data if d["category"] == "H2pisingerlowdim"])
    xiang_count = len([d for d in all_data if d["category"] == "H2xiang"])

    print("\nDone!")
    print("Generated file:")
    print(
        f"  - {output_file} ({large_scale_count} large-scale + {low_dim_count} low-dimensional + {xiang_count} Xiang instances)"
    )


if __name__ == "__main__":
    main()
