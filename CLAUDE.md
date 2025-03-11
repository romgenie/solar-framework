# SOLAR Framework Documentation

## Project Overview

SOLAR (Self-Optimized Logical and Arithmetic Reasoning) is a framework that uses different reasoning topologies for AI problem-solving. It implements multiple reasoning approaches (Chain of Thought, Tree of Thought, Graph of Thought) and dynamically selects the best approach based on problem characteristics.

## Benchmarking Commands

### Running Benchmarks

The main benchmark runner supports various options:

```bash
# Run both benchmarks with default settings (2 problems per category, 2 runs each)
./run_standard_benchmarks.py

# Run only the comprehensive benchmark
./run_standard_benchmarks.py --comprehensive --problems 3 --runs 2

# Run only the topological benchmark with a specific model
./run_standard_benchmarks.py --topological --problems 2 --runs 3 --ollama-model llama3.2
```

### Available Options

- `--comprehensive`: Run the comprehensive benchmark comparing different frameworks
- `--topological`: Run the topological benchmark comparing different reasoning topologies
- `--problems INT`: Number of problems to use per category (default: 2)
- `--runs INT`: Number of runs per problem (default: 2)
- `--ollama-model STRING`: Ollama model to use (default: llama3.2)

## Project Structure

```
/experiments/
├── CLAUDE.md                            # This documentation file
├── baseline_model.py                    # Control implementation without topology selection
├── datasets/                            # Standard benchmark datasets
│   ├── aqua_rat/                        # Algebra problems
│   ├── crt/                             # Cognitive Reflection Test problems
│   ├── gsm8k/                           # Grade School Math problems
│   └── logiqa/                          # Logical reasoning problems
├── download_datasets.sh                 # Script to download standard datasets
├── run_standard_benchmarks.py           # Main benchmark runner
├── solar_comprehensive_analysis.py      # Visualization and analysis for comprehensive benchmark
├── solar_comprehensive_benchmark.py     # Main benchmark across all approaches
├── solar_dataset_loader.py              # Loader for standard datasets
├── solar_evaluation_pipeline.py         # Metric computation and evaluation utilities
├── solar_hybrid_scaling.py              # Integration of fine-tuned models with topology
├── solar_main.py                        # Main entry point for all components
├── solar_ollama_benchmark.py            # Benchmark for Ollama models
├── solar_ollama_benchmark_analysis.py   # Analysis for Ollama benchmark results
├── solar_tag_system.py                  # Training data annotation and generation
├── solar_topological_benchmark.py       # Compare individual topologies against each other
└── solar_topological_rewarding.py       # Base reasoning topologies and reward model
```

## Core Components

1. **Reasoning Topologies**:
   - `ChainOfThought`: Linear sequential reasoning
   - `TreeOfThought`: Hierarchical branched reasoning
   - `GraphOfThought`: Network-based reasoning with interconnected nodes

2. **Reward Models**:
   - `MultiTaskTopologicalRewardModel`: Evaluates responses against ground truth answers

3. **Inference Pipelines**:
   - `InferencePipeline`: Basic topology-aware routing
   - `HybridScalingInference`: Combines fine-tuned models with topology selection

4. **Benchmark Datasets**:
   - GSM8K: Grade School Math problems
   - AQUA-RAT: Algebra Question Answering with Rationales
   - LogiQA: Logical reasoning problems
   - CRT: Cognitive Reflection Test

## Result Analysis

Benchmark results are saved to:
- `solar_topology_benchmark/`: For topological benchmark results
- `solar_analysis_results/`: For comprehensive benchmark results

Key metrics captured:
- Latency (inference time)
- Topology selection patterns
- Reward scores (quality assessment)
- Accuracy (comparison with ground truth)
- Success rate

## Development Guidelines

1. Always verify results against ground truth answers when available
2. When adding new problem types, follow the pattern in `solar_dataset_loader.py`
3. Run benchmarks with small samples first to ensure everything works
4. Use the `--problems` and `--runs` flags to control benchmark size

## Terminology

- **CoT**: Chain of Thought - sequential reasoning steps
- **ToT**: Tree of Thought - branching reasoning paths
- **GoT**: Graph of Thought - interconnected concept exploration
- **M-TRM**: Multi-task Topological Reward Model