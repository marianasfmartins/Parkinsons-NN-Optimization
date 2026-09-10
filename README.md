# Parkinson's Disease Neural Network Optimization

## Overview

This project applies **metaheuristic optimization algorithms** to optimize neural network weights for Parkinson's disease classification. The repository implements and compares two advanced optimization techniques: **Genetic Algorithm (GA)** and **Grey Wolf Optimizer (GWO)** to improve neural network performance on Parkinson's disease detection using the UCI Parkinson's dataset.

The project explores both standard optimization approaches and enhanced variants incorporating **L2 regularization** to prevent overfitting and promote simpler model architectures.

## Project Motivation

Traditional neural network training relies on gradient-based optimization (e.g., backpropagation). This project investigates whether metaheuristic algorithms can provide alternative or complementary optimization strategies for:
- Finding better weight configurations
- Reducing overfitting through regularization
- Exploring the solution space more effectively

## Dataset

- **Source**: Parkinson's Disease Classification Dataset (preprocessed)
- **Location**: `data/parkinsons_preprocessed.csv`
- **Features**: 21 input features (voice measurements and acoustic characteristics)
- **Target**: Binary classification (status: healthy vs. Parkinson's disease)
- **Data Split**:
  - Training: 60% (0.75 of the remaining data after 20% test split)
  - Validation: 15% (0.25 of the training data)
  - Test: 25%

## Architecture

### Neural Network
- **Type**: Multi-Layer Perceptron (MLP)
- **Input Layer**: 21 features
- **Hidden Layer**: 100 neurons with ReLU activation
- **Output Layer**: 1 neuron (binary classification)
- **Solver**: Adam optimizer (for comparison baseline)

### Total Network Parameters
- **Weights**: Approximately 2,100 (21×100 hidden + 100×1 output)
- **Biases**: Approximately 101 (100 hidden + 1 output)
- **Total Chromosome Length**: ~2,200 parameters to optimize

## Optimization Algorithms

### 1. Genetic Algorithm (GA)

**File**: `GA_main.py` and `algorithms.py`

A classic evolutionary algorithm inspired by natural selection:
- **Population Size**: 50 individuals
- **Generations**: 200
- **Selection**: Tournament selection (pool size = 3)
- **Crossover**: Blend crossover for real-valued vectors (probability = 0.8)
- **Mutation**: Gaussian mutation with σ = 0.05 (mutation probability = 0.1)
- **Elitism**: Best individual is preserved across generations

**Key Features**:
- Full generational replacement strategy
- Adaptive mutation strength
- Fitness history tracking for convergence analysis

### 2. Grey Wolf Optimizer (GWO)

**File**: `GWO_main.py` and `algorithms.py`

A swarm intelligence algorithm inspired by grey wolf pack hunting behavior:
- **Population Size (Pack)**: 30 wolves
- **Iterations**: 50
- **Leaders**: Alpha (best), Beta (second-best), Delta (third-best)
- **Exploration-Exploitation**: Linearly decreasing parameter `a` (2 → 0)

**Key Features**:
- Vectorized weight updates for computational efficiency
- Three-tier leadership hierarchy
- Adaptive balance between exploration and exploitation
- Weight update magnitude tracking for convergence diagnostics

### 3. L2-Regularized Variants

**Files**: `algorithms.py` (functions `genetic_algorithm_l2` and `grey_wolf_optimizer_l2`)

Enhanced versions incorporating L2 regularization:
- **Purpose**: Break ties when multiple solutions achieve equal classification error
- **Strategy**: Prefer solutions with smaller L2 norms (simpler models with smaller weights)
- **Formula**: L2 Norm = Σ(weights²)
- **Application**: Used as secondary fitness criterion when accuracy is equal

## Project Structure

```
Parkinsons-NN-Optimization/
├── README.md                          # Project documentation
├── algorithms.py                      # Core optimization algorithms (GA, GWO, L2 variants)
├── NN.py                              # Neural network interface & weight injection
├── project_data.py                    # Data loading and preprocessing
├── utils.py                           # Utility functions (fitness, visualization)
├── compare_algorithms.py              # Comparative analysis of all methods
├── GA_main.py                         # Genetic Algorithm execution script
├── GA_Randomized_Search.py            # GA variant with randomized parameter search
├── GWO_main.py                        # Grey Wolf Optimizer execution script
├── GWO_Greed_Search.py                # GWO variant with greedy parameter optimization
├── test_main.py                       # Testing and validation framework
├── testing_stuff.py                   # Additional testing utilities
├── GA_operators/                      # Genetic Algorithm operators module
│   └── population.py                  # Population initialization strategies
├── data/                              # Dataset directory
│   └── parkinsons_preprocessed.csv   # Parkinson's disease dataset
├── Project_OA.pdf                     # Detailed project report
├── OptimizerAlgorithmsReport.docx     # Technical documentation
└── *.png                              # Visualization outputs
    ├── GA_Fitness_Evolution.png       # GA convergence curve
    ├── GA_Fitness_Evolution_L2.png    # GA with L2 regularization
    ├── GWO_Fitness_Evolution.png      # GWO convergence curve
    ├── GWO_test_performance.png       # GWO test set performance
    └── metaheuristic_system_comparison.png
```

## Key Files

### `algorithms.py`
Core implementation of all four optimization variants:
- `genetic_algorithm()`: Standard GA for NN weight optimization
- `genetic_algorithm_l2()`: GA with L2 regularization tie-breaking
- `grey_wolf_optimizer()`: Standard GWO for NN weight optimization
- `grey_wolf_optimizer_l2()`: GWO with L2 regularization tie-breaking

### `NN.py`
- `get_predictions()`: Injects optimized weights into pre-trained MLP and generates predictions

### `project_data.py`
- Loads and partitions Parkinson's dataset
- Handles stratified train/validation/test splits

### `utils.py`
- `fitness_misclassification()`: Calculates misclassification error rate
- `plot_history()`: Generates convergence and performance visualizations

### Execution Scripts
- **`GA_main.py`**: Run standard genetic algorithm
- **`GWO_main.py`**: Run standard grey wolf optimizer
- **`GA_Randomized_Search.py`**: Hyperparameter optimization for GA
- **`GWO_Greed_Search.py`**: Hyperparameter optimization for GWO
- **`compare_algorithms.py`**: Side-by-side performance comparison

## Usage

### Basic Execution

#### Run Genetic Algorithm
```bash
python GA_main.py
```

#### Run Grey Wolf Optimizer
```bash
python GWO_main.py
```

#### Compare All Algorithms
```bash
python compare_algorithms.py
```

### Expected Output

Each algorithm will display:
- Initial fitness (generation/iteration 0)
- Iterative progress with best fitness improvements
- Final optimized weight vector
- Convergence plots saved as PNG files

Example GA output:
```
[GA] Generation 0/200 — Initial Best Fitness (Lowest Error): 0.125000
generation 1/200 — best fitness: 0.125000
generation 2/200 — best fitness: 0.095238
...
generation 200/200 — best fitness: 0.047619
```

## Results & Visualization

The project generates multiple visualizations:

1. **Fitness Evolution Curves**: Track best fitness over generations/iterations
2. **Weight Update Magnitudes**: Monitor convergence speed and stability
3. **Performance Comparison**: Side-by-side algorithm performance metrics
4. **L2 Regularization Impact**: Show how L2 variants affect convergence

### Sample Outputs
- `GA_Fitness_Evolution.png`: GA convergence trajectory
- `GA_Fitness_Evolution_L2.png`: GA with L2 regularization
- `GWO_Fitness_Evolution.png`: GWO convergence trajectory
- `GWO_test_performance.png`: Test set accuracy progression

## Hyperparameter Tuning

The project includes automated hyperparameter search scripts:

### `GA_Randomized_Search.py`
Randomized search over:
- Population size: 30-100
- Mutation probability: 0.05-0.3
- Crossover probability: 0.5-0.9
- Mutation strength: 0.01-0.1
- Number of generations: 100-300

### `GWO_Greed_Search.py`
Greedy search over:
- Pack size: 20-60
- Max iterations: 30-100

## Dependencies

```python
numpy                # Numerical computing
pandas               # Data manipulation
scikit-learn         # Machine learning models and metrics
matplotlib           # Plotting and visualization
scipy                # Scientific computing utilities
```

### Installation
```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

## Key Findings & Insights

### Algorithm Comparison
- **Genetic Algorithm**: Good exploration capability, slower convergence for NN weights
- **Grey Wolf Optimizer**: Faster convergence, effective for continuous optimization
- **L2 Regularization**: Improves generalization, prevents overfitting on tie solutions

### Convergence Behavior
- GA typically reaches plateau by generation 50-100
- GWO reaches near-optimal solutions faster (10-20 iterations)
- L2 variants show more stable final solutions with smaller weight magnitudes

### Classification Performance
- Both algorithms can outperform random initialization
- Validation accuracy typically improves 5-15% over baseline
- L2 variants often achieve better test set generalization

## Parameters Summary

| Parameter | GA | GWO |
|-----------|-----|-----|
| Population/Pack Size | 50 | 30 |
| Generations/Iterations | 200 | 50 |
| Selection Method | Tournament | Social Hierarchy |
| Mutation/Update | Gaussian | Triangular Envelope |
| Elitism/Preservation | Yes | Implicit (hierarchical) |

## Future Enhancements

1. **Multi-objective Optimization**: Optimize both accuracy and model complexity
2. **Hybrid Approaches**: Combine GA and GWO (genetic operators with swarm updates)
3. **Parallel Evaluation**: Distribute fitness evaluation across multiple workers
4. **Advanced Initialization**: Population diversity seeding strategies
5. **Transfer Learning**: Pre-trained weight vectors as initialization
6. **Adaptive Parameters**: Dynamic mutation rates and neighborhood sizes
7. **Extended Datasets**: Test on larger medical datasets
8. **Deep Networks**: Optimize weights for deeper architectures

## Project Reports

- **`Project_OA.pdf`**: Comprehensive technical report with theory and experiments
- **`OptimizerAlgorithmsReport.docx`**: Detailed documentation of all algorithms

## References

- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). "Grey Wolf Optimizer." Advances in Engineering Software.
- Mitchell, M. (1996). "An Introduction to Genetic Algorithms." MIT Press.
- UCI Machine Learning Repository: Parkinson's Disease Dataset
- Scikit-learn Documentation: Neural Network Classifiers

## License

This project is part of an academic study on optimization algorithms. Please refer to the original source repository for license information.

## Author

**Mariana SF Martins**

## Acknowledgments

This project builds upon the work in the parent repository: [Optimization_Algorithms](https://github.com/Maiara-Almada/Optimization_Algorithms)


