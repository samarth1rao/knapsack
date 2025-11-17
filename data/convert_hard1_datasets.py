#!/usr/bin/env python3
"""
Script to process knapsack problem instances and generate CSV datasets.
Processes all testcases from data/gh_knapsackProblemInstances/problemInstances directory.
"""

import csv
from pathlib import Path


def parse_test_input(test_in_path):
    """Parse test.in file and extract n, weights, prices, and capacity."""
    with open(test_in_path, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    weights = []
    prices = []

    for i in range(1, n + 1):
        parts = lines[i].strip().split()
        # parts[0] is item_id, which we don't need to store
        price = int(parts[1])
        weight = int(parts[2])
        weights.append(weight)
        prices.append(price)

    capacity = int(lines[n + 1].strip())

    return n, weights, prices, capacity


def parse_output(outp_out_path):
    """Parse outp.out file and extract best_price and best_picks."""
    try:
        with open(outp_out_path, "r") as f:
            lines = f.readlines()

        if (
            not lines
            or lines[0].strip().startswith("ERROR")
            or lines[0].strip().startswith("Error")
        ):
            return None, None

        # First line is the total profit
        best_price = int(lines[0].strip())

        # Subsequent lines contain item descriptions (profit weight)
        # We need to find which items from the original list match these
        selected_items = []
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    price = int(parts[0])
                    weight = int(parts[1])
                    selected_items.append((price, weight))

        return best_price, selected_items
    except (ValueError, FileNotFoundError):
        return None, None


def find_item_indices(weights, prices, selected_items):
    """Find the indices of selected items in the original lists."""
    indices = []
    used = [False] * len(weights)

    for sel_price, sel_weight in selected_items:
        # Find matching item that hasn't been used yet
        for i in range(len(weights)):
            if not used[i] and weights[i] == sel_weight and prices[i] == sel_price:
                indices.append(i)
                used[i] = True
                break

    return sorted(indices)


def load_optima(optima_path):
    """Load the optima.csv file and return a dictionary mapping name to optimum."""
    optima = {}
    with open(optima_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            optimum = int(row["optimum"])
            optima[name] = optimum
    return optima


def main():
    # Setup paths
    base_dir = Path(__file__).parent
    instances_dir = base_dir / "gh_knapsackProblemInstances" / "problemInstances"
    optima_path = base_dir / "gh_knapsackProblemInstances" / "optima.csv"

    output_file = base_dir / "knapsack_hard1_dataset.csv"

    # Load optima
    print("Loading optima.csv...")
    optima = load_optima(optima_path)
    print(f"Loaded {len(optima)} entries from optima.csv")

    # List to store all data
    all_data = []

    # Process all subdirectories
    print("\nProcessing problem instances...")
    instance_folders = sorted([d for d in instances_dir.iterdir() if d.is_dir()])

    for idx, folder in enumerate(instance_folders, 1):
        folder_name = folder.name
        test_in_path = folder / "test.in"
        outp_out_path = folder / "outp.out"

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(instance_folders)} instances...")

        if not test_in_path.exists():
            print(f"Warning: {folder_name} missing test.in")
            continue

        # Parse test input
        n, weights, prices, capacity = parse_test_input(test_in_path)

        # Check if optimum is known
        optimum = optima.get(folder_name, -1)

        if optimum != -1:
            # Parse output to get best picks
            best_price, selected_items = parse_output(outp_out_path)

            if best_price is not None and best_price == optimum:
                # Find indices of selected items
                best_picks = find_item_indices(weights, prices, selected_items)

                all_data.append(
                    {
                        "category": "H1known",
                        "n": n,
                        "weights": str(weights),
                        "prices": str(prices),
                        "capacity": capacity,
                        "best_picks": str(best_picks),
                        "best_price": best_price,
                    }
                )
            else:
                print(
                    f"Warning: {folder_name} - could not parse output or mismatch with optima"
                )
        else:
            # Optimum not known
            all_data.append(
                {
                    "category": "H1unknown",
                    "n": n,
                    "weights": str(weights),
                    "prices": str(prices),
                    "capacity": capacity,
                    "best_picks": "",
                    "best_price": "",
                }
            )

    print(f"\nProcessed {len(instance_folders)} instances")
    known_count = sum(1 for row in all_data if row["category"] == "H1known")
    unknown_count = sum(1 for row in all_data if row["category"] == "H1unknown")
    print(f"Known optima: {known_count}")
    print(f"Unknown optima: {unknown_count}")

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

    print("\nDone!")
    print("Generated file:")
    print(
        f"  - {output_file} ({known_count} known + {unknown_count} unknown instances)"
    )


if __name__ == "__main__":
    main()
