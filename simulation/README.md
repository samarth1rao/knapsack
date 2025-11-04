# Knapsack Algorithm Simulation

## Overview

The simulation system:
- Runs multiple knapsack algorithms on test datasets
- Collects performance metrics (time, memory, accuracy)
- Generates comprehensive visualizations
- Compares algorithm trade-offs

## Setup

### Prerequisites
- Python 3.7+
- C++ compiler (g++)
- Required Python packages (see requirements.txt)

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Compile algorithms:
```bash
cd ../algorithms
make all
```

## Usage

Run the simulation:
```bash
python simulate.py
```

## Metrics Collected

### 1. Execution Time
- Measured in microseconds
- Shows algorithm scalability

### 2. Solution Quality
- Compared against optimal solution
- Accuracy percentage calculated

### 3. Memory Usage
- Approximate memory footprint
- Important for large-scale problems

### 4. Optimality Gap
- For heuristic algorithms
- Shows trade-off between speed and quality

## Visualizations Generated

### 1. Time vs Problem Size
- Log-scale plot showing execution time growth
- Identifies algorithmic complexity

### 1b. Time vs Knapsack Capacity
- Scatter plot showing execution time against knapsack capacity
- Helps understand how capacity (not just item count) affects runtime

### 2. Quality vs Time (Pareto Plot)
- Trade-off between solution quality and speed
- Helps choose best algorithm for specific needs

### 3. Accuracy Distribution
- Box plots showing solution quality consistency
- Identifies reliable algorithms

### 4. Memory Usage
- Memory consumption across problem sizes
- Important for resource-constrained environments

### 5. Optimality Rate
- Percentage of optimal solutions found
- Key metric for exact algorithms

### 6. Summary Statistics Table
- Comprehensive performance overview
- Easy comparison across algorithms

## Output Structure

```
simulation/
├── simulate.py           # Main simulation script
├── results/              # CSV files with raw data
│   └── results_Tiny_YYYYMMDD_HHMMSS.csv
└── visualizations/       # Generated plots
    └── Tiny_YYYYMMDD_HHMMSS/
        ├── time_vs_size.png
    ├── time_vs_capacity.png
        ├── quality_vs_time.png
        ├── accuracy_distribution.png
        ├── memory_usage.png
        ├── optimality_rate.png
        └── summary_table.png
```

## Test Categories

- **Tiny**: 20-40 items (Verify exactness)
- **Small**: 10² - 10³ items (Test DP/B&B)
- **Medium**: 10⁴ - 10⁵ items (Test Heuristics)
- **Large**: 10⁶ - 10⁷ items (Test specialized algos)
- **Massive**: 10⁸ - 10⁹ items (Extreme scale)

## Adding New Algorithms

1. Implement algorithm in C++ (see bruteforce.cpp for I/O format)
2. Add compilation rule to algorithms/Makefile
3. Update algorithms dictionary in simulate.py
4. Run simulation

## Input/Output Format

### Input to Algorithm (stdin):
```
n capacity
w1 w2 w3 ... wn
v1 v2 v3 ... vn
```

### Output from Algorithm (stdout):
```
max_value
num_selected_items
item_index_1 item_index_2 ... item_index_k
execution_time_microseconds
memory_used_bytes
```

## Notes

- All indices are 0-based
- Execution time is measured internally by the algorithm
- Memory usage is approximate
- Algorithms should timeout after 5 minutes
 - simulate.py now uses an adaptive per-test timeout heuristic (base + per-item), capped by a maximum; see `simulate.py` for details
 - Time and memory visualizations use scatter plots (unconnected dots) to avoid misleading line connections for many random points
