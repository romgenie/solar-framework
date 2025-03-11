# Benchmark Datasets for SOLAR Framework

This directory contains the four standard benchmark datasets used to evaluate the SOLAR framework, as specified in the paper:

## Dataset Overview

1. **GSM8K** (Grade School Math Word Problems)
   - **Source**: https://github.com/openai/grade-school-math
   - **Size**: 8.5K high-quality math word problems
   - **Purpose**: Tests arithmetic reasoning capabilities
   - **Format**: JSON Lines with questions and step-by-step answers
   - **Files**: `train.jsonl`, `test.jsonl`
   - **Category in SOLAR**: "arithmetic"

2. **AQUA-RAT** (Algebra Question Answering with Rationales)
   - **Source**: https://github.com/deepmind/AQuA
   - **Size**: ~250 algebraic word problems
   - **Purpose**: Tests algebraic reasoning capabilities
   - **Format**: JSON with problems, rationales, and multiple-choice options
   - **Files**: `aqua.json`
   - **Category in SOLAR**: "algebra"

3. **LogiQA** (Logical Reasoning Questions)
   - **Source**: https://github.com/lgw863/LogiQA-dataset
   - **Size**: ~8K logical reasoning problems
   - **Purpose**: Tests deductive and inductive reasoning
   - **Format**: Text files with problems and multiple-choice options
   - **Files**: `train.txt`, `eval.txt`, `test.txt`
   - **Category in SOLAR**: "logic"

4. **CRT** (Cognitive Reflection Test)
   - **Source**: Manually created based on well-known CRT items
   - **Size**: 3 classic cognitive reflection problems
   - **Purpose**: Tests ability to override intuitive but incorrect answers
   - **Format**: JSON with problems, intuitive answers, and correct answers
   - **Files**: `crt_problems.json`
   - **Category in SOLAR**: "edge_cases"

## Dataset Examples

### GSM8K Example
```json
{
  "question": "Janet's ducks lay 16 eggs per day. She eats 3 eggs per day. How many eggs does she have after a week?",
  "answer": "Janet's ducks lay 16 eggs per day.\nSo in 7 days they lay 16 * 7 = 112 eggs.\nJanet eats 3 eggs per day.\nSo in 7 days she eats 3 * 7 = 21 eggs.\nSo she has 112 - 21 = 91 eggs after a week.\nThe answer is 91."
}
```

### AQUA-RAT Example
```json
{
  "question": "If 3x + y = 10 and x - 2y = 15, what is the value of x?",
  "options": ["7", "9", "12", "13", "15"],
  "rationale": "We have the system of equations: 3x + y = 10, x - 2y = 15...",
  "correct": "A"
}
```

### CRT Example
```json
{
  "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
  "intuitive_answer": "$0.10",
  "correct_answer": "$0.05",
  "explanation": "Let x be the cost of the ball. Then the bat costs x + $1.00. We have x + (x + $1.00) = $1.10, so 2x + $1.00 = $1.10, thus 2x = $0.10, and x = $0.05."
}
```

## Using the Datasets

### Downloading and Setup

To download and set up all datasets, run:
```bash
./download_datasets.sh
```

### Loading Datasets in SOLAR

The datasets are loaded by the `solar_dataset_loader.py` module:

```python
from solar_dataset_loader import DatasetLoader

# Initialize the loader
loader = DatasetLoader()

# Load arithmetic problems from GSM8K
gsm8k_problems = loader.load_gsm8k(sample_size=5)

# Load all datasets and organize by problem category
benchmark_problems = loader.load_benchmark_problems(sample_size=3)
```

### Verifying Results

The benchmark now includes verification of model answers against ground truth:

1. Model responses are scored based on similarity to ground truth answers
2. Accuracy metrics are reported for all benchmarks
3. Detailed answer comparison is included in analysis reports

## Adding New Datasets

To add a new dataset:

1. Create a new subdirectory in `datasets/`
2. Add the dataset files and a README.md explaining the format
3. Implement a loader method in `solar_dataset_loader.py`
4. Add the dataset to the `load_benchmark_problems()` method

For benchmark implementation examples, see:
- `solar_comprehensive_benchmark.py`
- `solar_topological_benchmark.py`

