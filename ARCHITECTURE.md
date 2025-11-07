# Knapsack Project Architecture

## System Overview

```txt
┌─────────────────────────────────────────────────────────────────┐
│                    KNAPSACK ANALYSIS SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│   DATA LAYER    │        │  ALGORITHM LAYER │        │  ANALYSIS LAYER  │
└─────────────────┘        └──────────────────┘        └──────────────────┘
        │                          │                            │
        ▼                          ▼                            ▼
┌─────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ Test Datasets   │───────>│ C++ Algorithms   │───────>│ Python Simulator │
│                 │        │                  │        │                  │
│ - Tiny (20-40)  │        │ ✓ Brute Force    │        │ - Run Tests      │
│ - Small (100+)  │        │ ✓ Memoization    │        │ - Collect Data   │
│ - Medium (10K+) │        │ ✓ Dynamic Prog   │        │ - Analyse        │
│ - Large (1M+)   │        │ ○ Branch & Bound │        │ - Visualise      │
│                 │        │ ✓ Meet in Middle │        │                  │
│ knapsack_       │        │ ✓ Greedy         │        │ simulate.py      │
│ dataset.csv     │        │ ✓ Random Perm    │        │                  │
└─────────────────┘        │ ○ Efficient      │        └──────────────────┘
                           │ ✓ Billion Scale  │                │
                           │ ✓ Genetic Algo   │                │
                           │                  │                ▼
                           │ bin/bruteforce   │        ┌──────────────────┐
                           └──────────────────┘        │  OUTPUT LAYER    │
                                   │                   └──────────────────┘
                                   │                           │
                                   ▼                           ▼
                           ┌──────────────────┐        ┌──────────────────┐
                           │   BUILD SYSTEM   │        │  Results & Viz   │
                           └──────────────────┘        └──────────────────┘
                           │                  │        │                  │
                           │ Makefile         │        │ - CSV Results    │
                           │ - Compile C++    │        │ - PNG Charts     │
                           │ - Link Binaries  │        │ - Statistics     │
                           │ - Clean Build    │        │                  │
                           └──────────────────┘        └──────────────────┘
```

## Data Flow Diagram

```txt
┌──────────────┐
│   CSV Data   │
│  (Datasets)  │
└──────┬───────┘
       │
       │ Read
       ▼
┌──────────────────┐
│ Python Simulator │
│                  │
│ 1. Parse CSV     │
│ 2. For each test │
│ 3. Prepare input │
└──────┬───────────┘
       │
       │ subprocess
       │ (WSL bridge)
       ▼
┌──────────────────┐
│ C++ Algorithm    │
│                  │
│ stdin:           │
│  n capacity      │
│  weights[]       │
│  values[]        │
│                  │
│ Process...       │
│                  │
│ stdout:          │
│  max_value       │
│  items[]         │
│  time            │
│  memory          │
└──────┬───────────┘
       │
       │ capture output
       ▼
┌──────────────────┐
│ Python Simulator │
│                  │
│ 4. Parse output  │
│ 5. Calc metrics  │
│ 6. Store results │
└──────┬───────────┘
       │
       │ After all tests
       ▼
┌──────────────────┐
│  Visualisation   │
│                  │
│ - matplotlib     │
│ - seaborn        │
│ - pandas         │
└──────┬───────────┘
       │
       │ Generate
       ▼
┌──────────────────┐
│ Output Files     │
│                  │
│ ✓ CSV results    │
│ ✓ PNG plots      │
│ ✓ Statistics     │
└──────────────────┘
```

## Technology Stack

```txt
┌─────────────────────────────────────────────┐
│              TECHNOLOGY STACK               │
└─────────────────────────────────────────────┘

Programming Languages:
  • C++ (Algorithms)        - Performance-critical code
  • Python (Simulation)     - Analysis & visualisation
  • Shell (Build)           - Automation

C++ Components:
  • Standard Library        - vectors, chrono, iostream
  • C++17 Features          - Modern syntax
  • G++ Compiler            - Optimisation flags

Python Libraries:
  • pandas                  - Data manipulation
  • numpy                   - Numerical operations
  • matplotlib              - Base plotting
  • seaborn                 - Statistical visualisation
  • subprocess              - Algorithm execution

Build Tools:
  • GNU Make                - Build automation
  • WSL                     - Windows/Linux bridge

Operating System:
  • Windows 10/11           - Development environment
  • WSL (Ubuntu)            - Unix compatibility
```

## Workflow

### Development Workflow

```txt
1. Implement Algorithm (C++)
   └─> algorithms/XX-name/algorithm.cpp

2. Update Makefile
   └─> Add compilation rule

3. Update Simulator
   └─> Add to algorithms dict in simulate.py

4. Test
   └─> make all
   └─> python simulate.py

5. Analyse
   └─> View visualisations
   └─> Compare with other algorithms
```

### Execution Workflow

```txt
User Run: python simulate.py
    │
    ├─> Load CSV Data
    │   └─> Parse test cases
    │
    ├─> For each test case:
    │   │
    │   ├─> Format input
    │   │   └─> "n capacity\nweights\nvalues"
    │   │
    │   ├─> Execute algorithm
    │   │   ├─> Windows Python
    │   │   └─> WSL C++ binary
    │   │
    │   ├─> Collect output
    │   │   └─> Parse results
    │   │
    │   └─> Calculate metrics
    │       ├─> Accuracy
    │       ├─> Time
    │       ├─> Memory
    │       └─> Optimality
    │
    ├─> Save Results
    │   └─> CSV file with timestamp
    │
    └─> Generate Visualisations
        ├─> Time analysis
   ├─> Time vs Capacity analysis
        ├─> Quality analysis
        ├─> Memory analysis
        ├─> Statistical summary
        └─> Save as PNG files
```

## Extensibility Points

### Adding New Algorithms

1. **Create Source File**

   ```sh
   algorithms/XX-name/algorithm.cpp
   ```

2. **Follow I/O Format**
   - Input: n, capacity, weights, values
   - Output: max_value, items, time, memory

3. **Update Makefile**

   ```makefile
   name: $(TARGET_DIR)/name
   $(TARGET_DIR)/name: XX-name/algorithm.cpp
       $(CXX) $(CXXFLAGS) -o $@ $<
   ```

4. **Register in Simulator**

   ```python
   self.algorithms = {
      "name": {
         "executable": <path>,
         "name": "<display name>",
         "sort_key": lambda n, w: <simplified big-O>, # <big-O formula>,
      }
   }
   ```

### Adding New Visualisations

1. **Create Method in KnapsackSimulator**

   ```python
   def _plot_new_viz(self, df, algorithms, viz_dir):
      # Create plot
      plt.savefig(viz_dir / "new_viz.png")
   ```

2. **Call in create_visualisations()**

   ```python
   self._plot_new_viz(results_df, algo_list, viz_dir)
   ```

### Adding New Metrics

1. **Extend Algorithm Output**
   - Add new metric to stdout

2. **Parse in run_algorithm()**
   - Extract from output lines

3. **Store in Results**
   - Add column to DataFrame

4. **Visualise**
   - Create appropriate plot

## Performance Considerations

### C++ Side

- Use `-O2` optimisation
- Minimise memory allocations
- Efficient recursion/iteration
- Measure time accurately (chrono)

### Python Side

- Batch operations with pandas
- Use vectorised numpy operations
- Generate plots once after all tests
- Save results incrementally

### WSL Bridge

- Minimise subprocess calls
- Batch input/output
- Handle timeouts gracefully
- Convert paths efficiently
