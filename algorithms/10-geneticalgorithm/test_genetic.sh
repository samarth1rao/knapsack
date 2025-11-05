#!/bin/bash

# Test script for genetic algorithm
# Uses the example from the Python code: 4 items with capacity 15

echo "Testing genetic algorithm with small example..."
echo "4 15
7 2 1 9
5 4 7 2" | ../bin/geneticalgorithm --population_size 50 --max_generations 100

echo -e "\n\nTesting with default parameters (matching Python)..."
echo "4 15
7 2 1 9
5 4 7 2" | ../bin/geneticalgorithm

echo -e "\n\nExpected optimal value: 16 (items B and C: values 4+7=11, weights 2+1=3)"
echo "Note: Genetic algorithm may not always find optimal solution"
