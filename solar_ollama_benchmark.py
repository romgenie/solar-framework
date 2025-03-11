"""
SOLAR Framework Benchmarking with Ollama

This script benchmarks the SOLAR framework components against Ollama models.
It measures performance, latency and quality metrics.
"""

import time
import json
import random
import requests
import statistics
from solar_topological_rewarding import InferencePipeline as SolarInferencePipeline
from solar_hybrid_scaling import HybridScalingInference
from solar_tag_system import TAGSystem
from solar_evaluation_pipeline import EvaluationPipeline

# Benchmark settings
NUM_PROBLEMS = 5
NUM_RUNS = 3
OLLAMA_MODELS = ["llama3.2", "deepseek-r1:8b"] 
BENCHMARK_PROBLEMS = [
    "Solve the math problem: What is 25 + 18?",
    "Solve the math problem: What is 14 * 6?",
    "Solve the math problem: What is the square root of 144?",
    "Solve the math problem: If 3x + 7 = 22, what is x?",
    "Solve the math problem: What is 40% of 85?"
]

def ollama_generate(model, prompt):
    """Generate response using Ollama API."""
    start_time = time.time()
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': model,
            'prompt': prompt,
            'stream': False
        }
    )
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        return {
            'response': result['response'],
            'latency': end_time - start_time
        }
    else:
        print(f"Error calling Ollama API: {response.status_code}")
        return {
            'response': "Error",
            'latency': 0
        }

def benchmark_solar_inference():
    """Benchmark the SOLAR Topological Rewarding inference pipeline."""
    pipeline = SolarInferencePipeline()
    results = []
    
    for problem in BENCHMARK_PROBLEMS:
        run_times = []
        for _ in range(NUM_RUNS):
            start_time = time.time()
            result = pipeline.process_request(problem)
            end_time = time.time()
            run_times.append(end_time - start_time)
        
        avg_time = statistics.mean(run_times)
        results.append({
            'problem': problem,
            'selected_topology': result['selected_topology'],
            'avg_latency': avg_time,
            'response': result['response']
        })
    
    return results

def benchmark_hybrid_scaling():
    """Benchmark the SOLAR Hybrid Scaling inference pipeline."""
    pipeline = HybridScalingInference()
    results = []
    
    for problem in BENCHMARK_PROBLEMS:
        run_times = []
        for _ in range(NUM_RUNS):
            start_time = time.time()
            result = pipeline.process_request(problem)
            end_time = time.time()
            run_times.append(end_time - start_time)
        
        avg_time = statistics.mean(run_times)
        results.append({
            'problem': problem,
            'selected_topology': result['selected_topology'],
            'avg_latency': avg_time,
            'response': result['response']
        })
    
    return results

def benchmark_ollama_models():
    """Benchmark Ollama models."""
    results = {}
    
    for model in OLLAMA_MODELS:
        model_results = []
        for problem in BENCHMARK_PROBLEMS:
            run_times = []
            run_responses = []
            
            for _ in range(NUM_RUNS):
                result = ollama_generate(model, problem)
                run_times.append(result['latency'])
                run_responses.append(result['response'])
            
            avg_time = statistics.mean(run_times)
            model_results.append({
                'problem': problem,
                'avg_latency': avg_time,
                'response': run_responses[0]  # Use first response for comparison
            })
        
        results[model] = model_results
    
    return results

def print_benchmark_results(solar_results, hybrid_results, ollama_results):
    """Print formatted benchmark results."""
    print("\n=== SOLAR Framework Benchmark Results ===\n")
    
    print("=== Latency Comparison (seconds) ===")
    print(f"{'Problem':<35} {'SOLAR':<10} {'Hybrid':<10}", end="")
    for model in OLLAMA_MODELS:
        print(f" {model:<15}", end="")
    print()
    
    for i, problem in enumerate(BENCHMARK_PROBLEMS):
        print(f"{problem[:35]:<35} {solar_results[i]['avg_latency']:<10.4f} {hybrid_results[i]['avg_latency']:<10.4f}", end="")
        for model in OLLAMA_MODELS:
            print(f" {ollama_results[model][i]['avg_latency']:<15.4f}", end="")
        print()
    
    print("\n=== Topology Selection Distribution ===")
    topology_counts = {}
    for result in solar_results:
        topo = result['selected_topology']
        topology_counts[topo] = topology_counts.get(topo, 0) + 1
    
    for topo, count in topology_counts.items():
        print(f"{topo}: {count} problems ({count/len(solar_results)*100:.1f}%)")
    
    print("\n=== Sample Responses ===")
    sample_idx = random.randint(0, len(BENCHMARK_PROBLEMS)-1)
    sample_problem = BENCHMARK_PROBLEMS[sample_idx]
    print(f"Problem: {sample_problem}")
    print(f"SOLAR: {solar_results[sample_idx]['response']}")
    print(f"Hybrid: {hybrid_results[sample_idx]['response']}")
    for model in OLLAMA_MODELS:
        print(f"{model}: {ollama_results[model][sample_idx]['response'][:150]}...")

if __name__ == "__main__":
    print("Starting SOLAR Framework benchmarking...")
    
    # Run benchmarks
    solar_results = benchmark_solar_inference()
    print("SOLAR Topological Rewarding benchmark complete.")
    
    hybrid_results = benchmark_hybrid_scaling()
    print("SOLAR Hybrid Scaling benchmark complete.")
    
    ollama_results = benchmark_ollama_models()
    print("Ollama models benchmark complete.")
    
    # Print results
    print_benchmark_results(solar_results, hybrid_results, ollama_results)
    
    # Save results to file
    with open('solar_benchmark_results.json', 'w') as f:
        json.dump({
            'solar': solar_results,
            'hybrid': hybrid_results,
            'ollama': ollama_results
        }, f, indent=2)
    
    print("\nBenchmark results saved to solar_benchmark_results.json")