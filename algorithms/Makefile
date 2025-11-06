# Makefile for Knapsack Algorithms
# Compatible with Unix/WSL

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
TARGET_DIR = bin

# Algorithm directories
ALGO_DIRS = 01-bruteforce 02-memoization 03-dynamicprogramming 04-branchandbound \
            05-meetinthemiddle 06-greedyheuristic 07-randompremutation 08-efficientalgo \
            09-billionscale 10-geneticalgorithm

# Explicit targets we currently build
TARGETS = $(TARGET_DIR)/bruteforce $(TARGET_DIR)/memoization $(TARGET_DIR)/dynamicprogramming $(TARGET_DIR)/randompermutation $(TARGET_DIR)/geneticalgorithm $(TARGET_DIR)/meetinthemiddle $(TARGET_DIR)/greedyheuristic

# Create bin directory if it doesn't exist
$(shell mkdir -p $(TARGET_DIR))

.PHONY: all clean bruteforce memoization dynamicprogramming randompermutation geneticalgorithm meetinthemiddle help
# Convenience single-target alias
meetinthemiddle: $(TARGET_DIR)/meetinthemiddle

$(TARGET_DIR)/meetinthemiddle: 05-meetinthemiddle/algo.cpp
	@echo "Compiling Meet in the Middle algorithm..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Meet in the Middle compiled successfully!"

# Default: build all explicit targets
all: $(TARGETS)

# Convenience single-target aliases
bruteforce: $(TARGET_DIR)/bruteforce
memoization: $(TARGET_DIR)/memoization
dynamicprogramming: $(TARGET_DIR)/dynamicprogramming
randompermutation: $(TARGET_DIR)/randompermutation
geneticalgorithm: $(TARGET_DIR)/geneticalgorithm
greedyheuristic: $(TARGET_DIR)/greedyheuristic

$(TARGET_DIR)/bruteforce: 01-bruteforce/bruteforce.cpp
	@echo "Compiling Brute Force algorithm..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Brute Force compiled successfully!"

$(TARGET_DIR)/memoization: 02-memoization/memoization.cpp
	@echo "Compiling Memoization algorithm..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Memoization compiled successfully!"

$(TARGET_DIR)/dynamicprogramming: 03-dynamicprogramming/dynamicprogramming.cpp
	@echo "Compiling Dynamic Programming algorithm..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Dynamic Programming compiled successfully!"

$(TARGET_DIR)/randompermutation: 07-randompremutation/randompermutation.cpp
	@echo "Compiling Random Permutation algorithm..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Random Permutation compiled successfully!"

$(TARGET_DIR)/geneticalgorithm: 10-geneticalgorithm/geneticalgorithm.cpp
	@echo "Compiling Genetic Algorithm..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Genetic Algorithm compiled successfully!"

$(TARGET_DIR)/greedyheuristic: 06-greedyheuristic/greedyheuristic.cpp
	@echo "Compiling Greedy Heuristic algorithm..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Greedy Heuristic compiled successfully!"

# Generic pattern rule (optional): compile any single-file cpp into bin/<basename>
$(TARGET_DIR)/%: %/%.cpp
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Compiled $@"

clean:
	@echo "Cleaning compiled binaries..."
	rm -rf $(TARGET_DIR)
	@echo "Clean complete!"

# Help target
help:
	@echo "Knapsack Algorithms Makefile"
	@echo "Usage:"
	@echo "  make all                    - Compile all explicit algorithms (bruteforce, memoization, dynamicprogramming, randompermutation, geneticalgorithm, greedyheuristic)"
	@echo "  make <name>                 - Compile a single algorithm, e.g. 'make bruteforce', 'make memoization', 'make dynamicprogramming', 'make randompermutation', 'make geneticalgorithm', or 'make greedyheuristic'"
	@echo "  make clean                  - Remove all compiled binaries"
	@echo "  make help                   - Display this help message"
	@echo "Notes: Add a new algorithm by placing its source in a numbered subdirectory (e.g. 11-newalgo/newalgo.cpp) and either add an explicit rule to this Makefile or rely on the generic pattern rule. The produced executable should be placed in $(TARGET_DIR) and named after the folder (or your chosen key)."
