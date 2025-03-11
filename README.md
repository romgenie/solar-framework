# SOLAR Framework: Self-Optimized Logical and Arithmetic Reasoning

SOLAR is a framework that dynamically selects the most appropriate reasoning topology (Chain of Thought, Tree of Thought, Graph of Thought) for AI problem-solving based on the problem type. This implementation includes benchmark tools that evaluate the framework against standard datasets.

## Overview

Different reasoning approaches work better for different types of problems. The SOLAR framework:

1. Uses multiple reasoning topologies:
   - **Chain of Thought (CoT)**: Sequential, step-by-step reasoning
   - **Tree of Thought (ToT)**: Branching, hierarchical reasoning
   - **Graph of Thought (GoT)**: Network-based, interconnected reasoning

2. Evaluates responses with a reward model that selects the best approach

3. Benchmarks performance against standard datasets:
   - GSM8K (arithmetic problems)
   - AQUA-RAT (algebraic problems)
   - LogiQA (logical reasoning problems)
   - CRT (cognitive reflection test problems)

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: matplotlib, pandas, seaborn
- For LLM benchmarks: Ollama with at least one model (default: llama3.2)

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd experiments
   ```

2. Download the benchmark datasets:
   ```bash
   bash download_datasets.sh
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running Benchmarks

Run benchmarks using the standard benchmark runner:

```bash
# Run both comprehensive and topological benchmarks
./run_standard_benchmarks.py

# Run only the comprehensive benchmark with more problems
./run_standard_benchmarks.py --comprehensive --problems 3

# Run only the topological benchmark with a specific model
./run_standard_benchmarks.py --topological --ollama-model llama2
```

### Options

- `--comprehensive`: Run comprehensive benchmark
- `--topological`: Run topological benchmark
- `--problems`: Number of problems per category (default: 2)
- `--runs`: Number of runs per problem (default: 2)
- `--ollama-model`: Model to use for LLM comparisons (default: llama3.2)

## Key Components

- `solar_topological_rewarding.py`: Core reasoning topologies and reward model
- `solar_hybrid_scaling.py`: Fine-tuned model integration
- `solar_dataset_loader.py`: Benchmark dataset handling
- `solar_comprehensive_benchmark.py`: Main benchmark tool
- `solar_topological_benchmark.py`: Topology comparison tool

## Results

Benchmark results are saved to:
- `solar_topology_benchmark/`: Analysis of different topologies
- `solar_analysis_results/`: Comprehensive benchmark results

Each results directory contains:
- JSON files with raw benchmark data
- PNG files with visualizations
- CSV files with answer comparisons
- Text reports summarizing findings

## Documentation

For more detailed information, see:
- `CLAUDE.md`: Main documentation file
- `datasets/README.md`: Information about benchmark datasets

## References

This implementation is based on research in reasoning approaches for language models:
- Chain of Thought: Wei et al. (2022)
- Tree of Thought: Yao et al. (2023)
- Graph of Thought: Besta et al. (2023)

## License

This project is licensed under the MIT License - see the LICENSE file for details.