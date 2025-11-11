"""Generate knapsack problems dataset using a C++ worker.

This script orchestrates the generation of a dataset of 0/1 knapsack problems.
It can generate either "easy" or "trap" problems via a command-line argument.

"Easy" problems: have a known optimal solution where the selected items have
much better price/weight ratios.

"Trap" problems: are hard by construction with a known optimal solution but
are designed to "trap" greedy (price/weight ratio) algorithms into finding
a sub-optimal one.

This script's roles:
1.  Parse command-line arguments (mode, total, level, seed, etc.).
2.  Compile the appropriate C++ worker program.
3.  Delete any old CSV and write a new header.
4.  Loop `total` times, calling the C++ worker for each problem.

The C++ worker's role:
1.  Receive params from this script.
        easy: category, n, seed
        trap: category, n, capacity, seed
2.  Generate a single instance.
3.  Append that single instance as a row to the CSV.

Command-line arguments (all provided as --arg value):
    --mode    : problem mode - "easy" or "trap" (default: trap)
    --out     : output CSV path (default: knapsack_easy_dataset.csv or knapsack_trap_dataset.csv resp.)
    --total   : total number of problems to generate (default: 100)
    --level   : maximum difficulty level to include (default: 2)
                    0 => Tiny only
                    1 => Tiny + Small
                    2 => Tiny + Small + Medium
                    3 => ONLY Large
                    4 => ONLY Massive
    --seed    : optional integer seed for reproducible sampling of (category,n)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import gc
import subprocess
import sys
from typing import Dict, List, Optional


def log_uniform_int(low: int, high: int, rng: random.Random) -> int:
    """Sample integer between low..high (inclusive) on a log-uniform scale."""
    if low >= high:
        return low
    log_low = math.log10(low)
    log_high = math.log10(high)
    u = rng.random() * (log_high - log_low) + log_low
    return max(low, min(high, int(round(10**u))))


def build_n_list(
    total: int, rng: random.Random, level: Optional[int] = None
) -> List[Dict]:
    """Build a list of (category, n) entries totaling `total` problems.

    Categories and sampling ranges:
      Tiny   : 20 - 40
      Small  : 10^2 - 10^3
      Medium : 10^4 - 10^5
      Large  : 10^6 - 10^7
      Massive: 10^8 - 10^9

    We distribute `total` roughly evenly across the included categories and
    sample n log-uniformly (Tiny uniformly).
    """
    categories = [
        ("Tiny", 20, 40),
        ("Small", 10**2, 10**3),
        ("Medium", 10**4, 10**5),
        ("Large", 10**6, 10**7),
        ("Massive", 10**8, 10**9),
    ]
    # level controls how many categories to include: 0..4
    if level is not None:
        if level < 0 or level > 4:
            raise ValueError("level must be 0, 1, 2, 3, 4, or omitted")
        if level <= 2:  # 0, 1, 2 are cumulative
            categories = categories[: level + 1]
        elif level == 3:  # Large only
            categories = [categories[3]]
        elif level == 4:  # Massive only
            categories = [categories[4]]

    # Distribute `total` evenly across the number of included categories.
    num_cats = len(categories)
    per_cat = [total // num_cats] * num_cats
    for i in range(total % num_cats):
        per_cat[i] += 1

    result = []
    # Use per-category counters to avoid rescanning `result` on every loop
    per_cat_counts = {c[0]: 0 for c in categories}
    for (cat, low, high), count in zip(categories, per_cat):
        # For very small ranges (like Tiny) it's possible the user requests more
        # entries than there are unique n values. In that case allow duplicates
        # instead of spinning forever trying to deduplicate.
        unique_possible = (high - low + 1) if cat == "Tiny" else None
        seen = set()
        attempts = 0
        while per_cat_counts[cat] < count and attempts < max(1000, count * 10):
            attempts += 1
            if cat == "Tiny":
                n = rng.randint(low, high)
            else:
                n = log_uniform_int(low, high, rng)
            # If duplicates are possible but the requested count exceeds the
            # unique possibilities, don't deduplicate.
            if unique_possible is not None and count <= unique_possible:
                if (cat, n) in seen:
                    continue
                seen.add((cat, n))
            result.append({"category": cat, "n": int(n)})
            per_cat_counts[cat] += 1

    # Shuffle to avoid grouped categories
    rng.shuffle(result)
    return result


def compile_cpp_worker(cpp_src: str, cpp_exe: str) -> bool:
    """Compile the C++ worker if it's missing or outdated."""
    # Ensure bin directory exists
    bin_dir = os.path.dirname(cpp_exe)
    if bin_dir:
        os.makedirs(bin_dir, exist_ok=True)

    if not os.path.exists(cpp_exe) or (
        os.path.exists(cpp_src)
        and os.path.getmtime(cpp_src) > os.path.getmtime(cpp_exe)
    ):
        print(f"Compiling C++ worker: {cpp_src} -> {cpp_exe}")
        compile_cmd = ["g++", "-O3", "-std=c++17", cpp_src, "-o", cpp_exe]

        try:
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
            print("Compilation successful.")
            return True
        except FileNotFoundError:
            print("Error: g++ compiler not found. Please install g++.", file=sys.stderr)
            return False
        except subprocess.CalledProcessError as e:
            print("Error: C++ compilation failed.", file=sys.stderr)
            print("STDOUT:", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
            print("STDERR:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            return False
    else:
        # print(f"C++ worker '{cpp_exe}' is up to date.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["easy", "trap"],
        default="trap",
        help="Problem mode: 'easy' or 'trap' (default: trap)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: data/knapsack_trap_dataset.csv or data/knapsack_easy_dataset.csv)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=100,
        help="Total number of problems to generate (default 100)",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=2,
        help="Which difficulty levels to include (default 2)\n",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Set default output path based on mode if not specified
    if args.out is None:
        if args.mode == "trap":
            args.out = os.path.join(
                os.path.dirname(__file__), "knapsack_trap_dataset.csv"
            )
        else:
            args.out = os.path.join(
                os.path.dirname(__file__), "knapsack_easy_dataset.csv"
            )

    # --- 1. Compile C++ Worker ---
    if args.mode == "trap":
        cpp_src_file = "generate_trap_instance.cpp"
        cpp_exe_file = "generate_trap_instance"
    else:
        cpp_src_file = "generate_easy_instance.cpp"
        cpp_exe_file = "generate_easy_instance"

    if sys.platform == "win32":
        cpp_exe_file += ".exe"

    script_dir = os.path.dirname(__file__)
    cpp_src_path = os.path.join(script_dir, cpp_src_file)
    if not os.path.exists(cpp_src_path):
        print(
            f"Error: C++ source file '{cpp_src_file}' not found in the same directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    bin_dir = os.path.join(script_dir, "bin")
    cpp_exe_path = os.path.join(bin_dir, cpp_exe_file)
    if not compile_cpp_worker(cpp_src_path, cpp_exe_path):
        sys.exit(1)

    # --- 2. Build Problem Specs ---
    rng = random.Random(args.seed)
    specs = build_n_list(args.total, rng, level=args.level)
    if not specs:
        print("No problems to generate. Exiting.")
        return

    # --- 3. Initialize CSV File and Header ---
    fields = [
        "category",
        "n",
        "weights",
        "prices",
        "capacity",
        "best_picks",
        "best_price",
        "seed",
    ]

    # Ensure output directory exists
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        if os.path.exists(args.out):
            os.remove(args.out)

        with open(args.out, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(fields) + "\n")
    except IOError as e:
        print(f"Error: Could not write to output file {args.out}. {e}", file=sys.stderr)
        sys.exit(1)

    # --- 4. Main Generation Loop (Call C++) ---
    mode_str = "trap" if args.mode == "trap" else "easy"
    print(f"Generating {len(specs)} {mode_str} problem instances into {args.out}...")
    for i, spec in enumerate(specs):
        cat = spec["category"]
        n = spec["n"]
        seed = rng.randint(0, 2**31 - 1)

        # Build command based on mode
        if args.mode == "trap":
            capacity = 0
            # Trap mode requires capacity parameter
            if cat == "Tiny":
                capacity = 100_000
            elif cat == "Small":
                capacity = 1_000_000
            elif cat == "Medium":
                capacity = 10_000_000
            elif cat == "Large":
                capacity = 100_000_000
            elif cat == "Massive":
                capacity = 1_000_000_000
            cmd = [cpp_exe_path, args.out, cat, str(n), str(capacity), str(seed)]
        else:
            # Easy mode does not require capacity
            cmd = [cpp_exe_path, args.out, cat, str(n), str(seed)]

        try:
            # We don't need to capture output, just check for errors
            subprocess.run(cmd, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(
                f"\nError: C++ worker failed for instance {i + 1} (n={n}, seed={seed})",
                file=sys.stderr,
            )
            print("STDERR:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            print("Aborting.", file=sys.stderr)
            sys.exit(1)

        # Simple progress bar
        print(f"  ... Wrote problem {i + 1}/{len(specs)} (n={n}, cat={cat})", end="\r")

        # Hint to GC
        if (i + 1) % 100 == 0:
            gc.collect()

    print(
        f"\nSuccessfully wrote {len(specs)} problems to {args.out} (mode={args.mode}, seed={args.seed}, level={args.level})"
    )


if __name__ == "__main__":
    main()
