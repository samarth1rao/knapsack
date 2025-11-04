# Algorithms for 0-1 Knapsack

We present 10 pseudopolynomial time algorithms which attempt to solve the classic 0-1 Knapsack problem.  
We also present a theoretical analysis of each of these algorithms.  
 
Please ensure that the input and output formatting is uniform so the simulation harness can run any algorithm without per-algorithm parsing changes.

## Build

- The repository includes a `Makefile` in this folder. Compiled executables are placed in `algorithms/bin/`.
- From the top of this folder you can run (unix/WSL):
	- `make all` to build the primary algorithms
	- `make <name>` to build a single algorithm (e.g. `make bruteforce`)

When adding a new algorithm, place the source file in a numbered subdirectory (for example `11-newalgo/newalgo.cpp`) and either add an explicit rule to the `Makefile` or rely on the generic pattern rule which compiles `subdir/subdir.cpp` to `bin/subdir`.

## How simulate.py runs algorithms

- `simulate.py` expects algorithm executables to be in `algorithms/bin/` and referenced by name in the simulator's `algorithms` dictionary. Each entry should contain the key used by the simulator and the path to the executable (absolute or relative).

## Input / Output Format (required)

Algorithms must read from standard input (stdin) and write to standard output (stdout) using the exact formats below so the simulator can feed inputs and parse outputs reliably.

Input to algorithm (stdin):
- First line: two integers separated by whitespace: `n capacity`
	- `n` = number of items (integer)
	- `capacity` = knapsack capacity (integer)
- Second line: `n` integers separated by whitespace: the item weights `w1 w2 ... wn`
- Third line: `n` integers separated by whitespace: the item values `v1 v2 ... vn`

Output from algorithm (stdout):
- First line: a single integer `max_value` — the best achievable total value.
- Second line: a single integer `num_selected_items` — the number of items selected in the reported solution.
- Third line: zero or more integers separated by spaces: the 0-based indices of the selected items `item_index_1 item_index_2 ... item_index_k`. If `num_selected_items` is zero the line may be empty or omitted.
- Fourth line: a single integer `execution_time_microseconds` — the algorithm's measured execution time in microseconds (integer). If the program does not measure time internally, the simulator will substitute a measured wall-clock time.
- Fifth line: a single integer `memory_used_bytes` — an approximate memory footprint in bytes (integer). If the program does not provide this, the simulator will substitute an estimated value.

Notes:
- All indices are 0-based.
- Each numeric output should be printable as a plain integer with no additional text, labels, or ANSI coloring so the simulator can parse the lines directly.
- The order of the selected item indices does not matter, but they must be unique and within `[0, n-1]`.

Following these rules ensures `simulate.py` can compile, execute, parse, and visualize results for any algorithm implemented here.
