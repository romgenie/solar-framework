#!/usr/bin/env python3
"""
SOLAR Framework Standard Benchmark Runner

This script runs the standard benchmarks using datasets from:
- GSM8K (Grade School Math)
- AQUA-RAT (Algebra Question Answering with Rationales)
- LogiQA (Logical Reasoning)
- CRT (Cognitive Reflection Test)

It provides options to control which benchmarks to run and how many problems to use.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SOLAR framework benchmarks with standard datasets")
    
    parser.add_argument("--comprehensive", action="store_true", 
                        help="Run comprehensive benchmark comparing different frameworks")
    parser.add_argument("--topological", action="store_true", 
                        help="Run topological benchmark comparing different reasoning topologies")
    parser.add_argument("--problems", type=int, default=2,
                        help="Number of problems to use per category (default: 2)")
    parser.add_argument("--runs", type=int, default=2,
                        help="Number of runs per problem (default: 2)")
    parser.add_argument("--ollama-model", type=str, default="llama3.2",
                        help="Ollama model to use (default: llama3.2)")
    
    args = parser.parse_args()
    
    # If no specific benchmark is selected, run both
    if not args.comprehensive and not args.topological:
        args.comprehensive = True
        args.topological = True
        
    return args

def run_comprehensive_benchmark(num_problems, num_runs, ollama_model):
    """Run the comprehensive benchmark."""
    print("=" * 70)
    print(f"Running Comprehensive Benchmark with {num_problems} problems per category")
    print("=" * 70)
    
    # Create a temporary modified version of solar_comprehensive_benchmark.py
    with open("solar_comprehensive_benchmark.py", "r") as f:
        content = f.read()
    
    # Modify the parameters
    modified_content = content.replace("NUM_RUNS = 2", f"NUM_RUNS = {num_runs}")
    modified_content = modified_content.replace("PROBLEMS_PER_CATEGORY = 2", f"PROBLEMS_PER_CATEGORY = {num_problems}")
    modified_content = modified_content.replace('OLLAMA_MODELS = [\n    "llama3.2"\n]', 
                                               f'OLLAMA_MODELS = [\n    "{ollama_model}"\n]')
    
    # Write to a temporary file
    with open("_temp_comprehensive_benchmark.py", "w") as f:
        f.write(modified_content)
    
    try:
        # Run the benchmark
        result = subprocess.run(["python", "_temp_comprehensive_benchmark.py"], 
                                capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Error running comprehensive benchmark: {e}")
    finally:
        # Clean up
        if os.path.exists("_temp_comprehensive_benchmark.py"):
            os.remove("_temp_comprehensive_benchmark.py")

def run_topological_benchmark(num_problems, num_runs, ollama_model):
    """Run the topological benchmark."""
    print("=" * 70)
    print(f"Running Topological Benchmark with {num_problems} problems per category")
    print("=" * 70)
    
    # Create a temporary modified version of solar_topological_benchmark.py
    with open("solar_topological_benchmark.py", "r") as f:
        content = f.read()
    
    # Modify the parameters
    modified_content = content.replace("NUM_RUNS = 2", f"NUM_RUNS = {num_runs}")
    modified_content = modified_content.replace("PROBLEMS_PER_CATEGORY = 1", f"PROBLEMS_PER_CATEGORY = {num_problems}")
    modified_content = modified_content.replace('OLLAMA_MODEL = "llama3.2"', f'OLLAMA_MODEL = "{ollama_model}"')
    
    # Write to a temporary file
    with open("_temp_topological_benchmark.py", "w") as f:
        f.write(modified_content)
    
    try:
        # Run the benchmark
        result = subprocess.run(["python", "_temp_topological_benchmark.py"], 
                                capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Error running topological benchmark: {e}")
    finally:
        # Clean up
        if os.path.exists("_temp_topological_benchmark.py"):
            os.remove("_temp_topological_benchmark.py")

def main():
    args = parse_args()
    
    print(f"SOLAR Framework Standard Benchmark Runner")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Settings:")
    print(f"  Problems per category: {args.problems}")
    print(f"  Runs per problem: {args.runs}")
    print(f"  Ollama model: {args.ollama_model}")
    print(f"  Comprehensive benchmark: {'Yes' if args.comprehensive else 'No'}")
    print(f"  Topological benchmark: {'Yes' if args.topological else 'No'}")
    print("\n")
    
    if args.comprehensive:
        run_comprehensive_benchmark(args.problems, args.runs, args.ollama_model)
        
    if args.topological:
        run_topological_benchmark(args.problems, args.runs, args.ollama_model)
    
    print("\nAll benchmarks completed!")

if __name__ == "__main__":
    main()