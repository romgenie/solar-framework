"""
Comprehensive SOLAR Framework Benchmark

This script conducts an extensive benchmark of the SOLAR framework against a variety of LLMs
across diverse problem categories, edge cases, and metrics.

The benchmark aims to evaluate:
1. Performance across different types of reasoning problems
2. Topology selection patterns for different problem categories
3. Comparative performance vs. various LLM models available via Ollama
4. Edge case handling and robustness
5. Quality metrics beyond simple latency

Uses standard benchmark datasets:
- GSM8K (Grade School Math)
- AQUA-RAT (Algebra Question Answering with Rationales)
- LogiQA (Logical Reasoning)
- CRT (Cognitive Reflection Test)
"""

import time
import json
import random
import string
import statistics
import requests
import concurrent.futures
import argparse
from datetime import datetime
from solar_topological_rewarding import InferencePipeline as SolarInferencePipeline
from solar_topological_rewarding import MultiTaskTopologicalRewardModel
from solar_hybrid_scaling import HybridScalingInference
from solar_tag_system import TAGSystem
from solar_evaluation_pipeline import EvaluationPipeline
from baseline_model import BaselineModel, SimplePromptTemplate
from solar_dataset_loader import DatasetLoader

# Test parameters - configurable
NUM_RUNS = 2
PROBLEMS_PER_CATEGORY = 2

# Test on just one model for demo purposes
OLLAMA_MODELS = [
    "llama3.2"
]

# Load benchmark problems from standard datasets
loader = DatasetLoader()
BENCHMARK_PROBLEMS = loader.load_benchmark_problems(sample_size=PROBLEMS_PER_CATEGORY)

# Fallback test problems (used if dataset loading fails)
FALLBACK_PROBLEMS = {
    "arithmetic": [
        "Solve the math problem: What is 25 + 18?",
        "Solve the math problem: What is 14 * 6?",
    ],
    "algebra": [
        "Solve the math problem: If 3x + 7 = 22, what is x?",
        "Solve the math problem: Solve for y in the equation 2y - 5 = 11.",
    ],
    "logic": [
        "Solve this logic problem: All roses are flowers. Some flowers fade quickly. Therefore, some roses fade quickly. Is this conclusion valid?",
        "Solve this logic problem: If it's raining, then the ground is wet. The ground is wet. Does this mean it's raining?",
    ],
    "edge_cases": [
        "Solve this ambiguous problem: If a bat and a ball cost $1.10 in total, and the bat costs $1.00 more than the ball, how much does the ball cost?",
        "Solve this problem with irrelevant information: John is 25 years old. Mary is 5 years older than John. John has a dog named Spot who is 3 years old. John's favorite color is blue. How old is Mary?",
    ]
}

# Use fallback problems for any category with empty problems
for category in BENCHMARK_PROBLEMS:
    if not BENCHMARK_PROBLEMS[category]:
        print(f"Warning: No problems loaded for {category}, using fallback problems")
        BENCHMARK_PROBLEMS[category] = FALLBACK_PROBLEMS[category]

def ollama_generate(model, prompt):
    """Generate response using Ollama API."""
    start_time = time.time()
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            },
            timeout=60  # Add timeout to prevent hanging
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            return {
                'response': result['response'],
                'latency': end_time - start_time,
                'status': 'success'
            }
        else:
            print(f"Error calling Ollama API: {response.status_code}")
            return {
                'response': f"Error: HTTP {response.status_code}",
                'latency': 0,
                'status': 'error'
            }
    except Exception as e:
        print(f"Exception when calling Ollama API: {str(e)}")
        return {
            'response': f"Error: {str(e)}",
            'latency': 0,
            'status': 'error'
        }

def benchmark_solar_inference(problem_category, problems):
    """Benchmark the SOLAR Topological Rewarding inference pipeline."""
    pipeline = SolarInferencePipeline()
    results = []
    
    # Load the full problems with answers
    loader = DatasetLoader()
    problem_details = {}
    
    # Load answers for each problem category
    if problem_category == "arithmetic":
        full_problems = loader.load_gsm8k(sample_size=PROBLEMS_PER_CATEGORY * 2)
        for p in full_problems:
            problem_details[p['problem']] = p
    elif problem_category == "algebra":
        full_problems = loader.load_aqua_rat(sample_size=PROBLEMS_PER_CATEGORY * 2)
        for p in full_problems:
            problem_details[p['problem']] = p
    elif problem_category == "logic":
        full_problems = loader.load_logiqa(sample_size=PROBLEMS_PER_CATEGORY * 2)
        for p in full_problems:
            problem_details[p['problem']] = p
    elif problem_category == "edge_cases":
        full_problems = loader.load_crt()
        for p in full_problems:
            problem_details[p['problem']] = p
    
    for problem in problems:
        run_times = []
        topology_selections = []
        accuracies = []
        extracted_answers = []
        
        # Find the ground truth for this problem
        ground_truth = None
        if problem in problem_details:
            if 'answer' in problem_details[problem]:
                ground_truth = problem_details[problem]['answer']
            elif 'correct_answer' in problem_details[problem]:
                ground_truth = problem_details[problem]['correct_answer']
        
        for _ in range(NUM_RUNS):
            start_time = time.time()
            result = pipeline.process_request(problem, ground_truth)
            end_time = time.time()
            
            run_times.append(end_time - start_time)
            topology_selections.append(result['selected_topology'])
            
            if 'accuracy' in result:
                accuracies.append(result['accuracy'])
                
            if 'extracted_answer' in result:
                extracted_answers.append(result['extracted_answer'])
        
        avg_time = statistics.mean(run_times)
        avg_accuracy = statistics.mean(accuracies) if accuracies else 0.0
        
        # Get most common topology selection
        most_common_topology = max(set(topology_selections), key=topology_selections.count)
        
        # Get most consistent extracted answer
        most_common_answer = None
        if extracted_answers:
            answer_counts = {}
            for answer in extracted_answers:
                if answer not in answer_counts:
                    answer_counts[answer] = 0
                answer_counts[answer] += 1
            most_common_answer = max(answer_counts, key=answer_counts.get)
        
        results.append({
            'problem': problem,
            'category': problem_category,
            'ground_truth': ground_truth,
            'extracted_answer': most_common_answer,
            'accuracy': avg_accuracy,
            'selected_topology': most_common_topology,
            'topology_distribution': {t: topology_selections.count(t)/NUM_RUNS for t in set(topology_selections)},
            'avg_latency': avg_time,
            'response': result['response'],
            'reward_score': result['score']
        })
    
    return results

def benchmark_hybrid_scaling(problem_category, problems):
    """Benchmark the SOLAR Hybrid Scaling inference pipeline."""
    pipeline = HybridScalingInference()
    results = []
    
    # Load the full problems with answers
    loader = DatasetLoader()
    problem_details = {}
    
    # Load answers for each problem category
    if problem_category == "arithmetic":
        full_problems = loader.load_gsm8k(sample_size=PROBLEMS_PER_CATEGORY * 2)
        for p in full_problems:
            problem_details[p['problem']] = p
    elif problem_category == "algebra":
        full_problems = loader.load_aqua_rat(sample_size=PROBLEMS_PER_CATEGORY * 2)
        for p in full_problems:
            problem_details[p['problem']] = p
    elif problem_category == "logic":
        full_problems = loader.load_logiqa(sample_size=PROBLEMS_PER_CATEGORY * 2)
        for p in full_problems:
            problem_details[p['problem']] = p
    elif problem_category == "edge_cases":
        full_problems = loader.load_crt()
        for p in full_problems:
            problem_details[p['problem']] = p
    
    for problem in problems:
        run_times = []
        topology_selections = []
        extracted_answers = []
        
        # Find the ground truth for this problem
        ground_truth = None
        if problem in problem_details:
            if 'answer' in problem_details[problem]:
                ground_truth = problem_details[problem]['answer']
            elif 'correct_answer' in problem_details[problem]:
                ground_truth = problem_details[problem]['correct_answer']
        
        for _ in range(NUM_RUNS):
            start_time = time.time()
            # Note: We assume HybridScalingInference has been updated to support ground_truth
            # If not, this will just ignore the parameter
            try:
                result = pipeline.process_request(problem, ground_truth) 
            except TypeError:
                # Fallback if ground_truth parameter is not supported
                result = pipeline.process_request(problem)
            end_time = time.time()
            
            run_times.append(end_time - start_time)
            topology_selections.append(result['selected_topology'])
            
            # Extract answer if it wasn't already done
            if 'extracted_answer' in result:
                extracted_answers.append(result['extracted_answer'])
            elif ground_truth and 'response' in result:
                # Create a temporary reward model to extract and verify the answer
                temp_model = MultiTaskTopologicalRewardModel()
                extracted_answer = temp_model.extract_answer(result['response'])
                extracted_answers.append(extracted_answer)
        
        avg_time = statistics.mean(run_times)
        
        # Get most common topology selection
        most_common_topology = max(set(topology_selections), key=topology_selections.count)
        
        # Get most consistent extracted answer
        most_common_answer = None
        if extracted_answers:
            answer_counts = {}
            for answer in extracted_answers:
                if answer not in answer_counts:
                    answer_counts[answer] = 0
                answer_counts[answer] += 1
            most_common_answer = max(answer_counts, key=answer_counts.get)
        
        # Calculate accuracy if ground truth is available
        accuracy = 0.0
        if ground_truth and most_common_answer:
            temp_model = MultiTaskTopologicalRewardModel()
            accuracy = temp_model.compute_string_similarity(most_common_answer, ground_truth)
        
        results.append({
            'problem': problem,
            'category': problem_category,
            'ground_truth': ground_truth,
            'extracted_answer': most_common_answer,
            'accuracy': accuracy,
            'selected_topology': most_common_topology,
            'topology_distribution': {t: topology_selections.count(t)/NUM_RUNS for t in set(topology_selections)},
            'avg_latency': avg_time,
            'response': result['response'],
            'reward_score': result['score']
        })
    
    return results

def benchmark_ollama_models(problem_category, problems):
    """Benchmark Ollama models."""
    results = {}
    
    for model in OLLAMA_MODELS:
        model_results = []
        for problem in problems:
            run_times = []
            run_responses = []
            success_count = 0
            
            for _ in range(NUM_RUNS):
                result = ollama_generate(model, problem)
                if result['status'] == 'success':
                    run_times.append(result['latency'])
                    run_responses.append(result['response'])
                    success_count += 1
            
            # Only calculate stats if we have successful runs
            if success_count > 0:
                avg_time = statistics.mean(run_times) if run_times else 0
                response = run_responses[0] if run_responses else "No successful responses"
                
                # Simple answer extraction (basic approach)
                answer = extract_answer(response)
                
                model_results.append({
                    'problem': problem,
                    'category': problem_category,
                    'avg_latency': avg_time,
                    'response': response,
                    'extracted_answer': answer,
                    'success_rate': success_count / NUM_RUNS
                })
            else:
                model_results.append({
                    'problem': problem,
                    'category': problem_category,
                    'avg_latency': 0,
                    'response': "All attempts failed",
                    'extracted_answer': None,
                    'success_rate': 0
                })
        
        results[model] = model_results
    
    return results

def extract_answer(response_text):
    """
    Simple heuristic to extract a final answer from model responses.
    This is a very basic implementation that could be improved with more sophisticated methods.
    """
    # Look for common answer indicators
    answer_indicators = [
        "The answer is", "Therefore,", "Thus,", "So,", "Final answer:",
        "Therefore the answer is", "In conclusion,", "The result is"
    ]
    
    # Try to find the answer after these indicators
    for indicator in answer_indicators:
        if indicator in response_text:
            answer_part = response_text.split(indicator, 1)[1].strip()
            # Take just the first sentence of the answer part
            if '.' in answer_part:
                return answer_part.split('.', 1)[0].strip()
            else:
                return answer_part.strip()[:100]  # Limit length
    
    # If no indicators found, take the last sentence
    sentences = response_text.split('.')
    if len(sentences) > 1:
        return sentences[-2].strip()  # Often the last sentence is just empty or a signature
    
    # Fallback: just return a truncated version of the full text
    return response_text[:100] + "..."

def benchmark_baseline_models(problem_category, problems):
    """Benchmark baseline models (direct LLM without topology selection)."""
    results = {}
    
    for model_name in OLLAMA_MODELS:
        baseline_model = BaselineModel(model_name=model_name)
        model_results = []
        
        for problem in problems:
            run_times = []
            run_responses = []
            success_count = 0
            
            formatted_prompt = SimplePromptTemplate.format_problem(problem)
            
            for _ in range(NUM_RUNS):
                result = baseline_model.process_request(formatted_prompt)
                if result['success']:
                    run_times.append(result['latency'])
                    run_responses.append(result['response'])
                    success_count += 1
            
            # Only calculate stats if we have successful runs
            if success_count > 0:
                avg_time = statistics.mean(run_times) if run_times else 0
                response = run_responses[0] if run_responses else "No successful responses"
                
                # Extract answer
                answer = extract_answer(response)
                
                model_results.append({
                    'problem': problem,
                    'category': problem_category,
                    'avg_latency': avg_time,
                    'response': response,
                    'extracted_answer': answer,
                    'success_rate': success_count / NUM_RUNS
                })
            else:
                model_results.append({
                    'problem': problem,
                    'category': problem_category,
                    'avg_latency': 0,
                    'response': "All attempts failed",
                    'extracted_answer': None,
                    'success_rate': 0
                })
        
        results[model_name] = model_results
    
    return results

def run_benchmarks():
    """Run all benchmarks."""
    results = {
        'solar': [],
        'hybrid': [],
        'ollama': {},  # Standard models with topology selection
        'baseline': {},  # Control group: direct LLM without topology
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_runs': NUM_RUNS,
            'models_tested': OLLAMA_MODELS
        }
    }
    
    # Run benchmarks for each problem category
    for category, problems in BENCHMARK_PROBLEMS.items():
        print(f"\nBenchmarking {category} problems...")
        
        # Run SOLAR benchmarks
        print(f"  Running SOLAR Topological Rewarding benchmark...")
        solar_results = benchmark_solar_inference(category, problems)
        results['solar'].extend(solar_results)
        
        print(f"  Running SOLAR Hybrid Scaling benchmark...")
        hybrid_results = benchmark_hybrid_scaling(category, problems)
        results['hybrid'].extend(hybrid_results)
        
        print(f"  Running Ollama models benchmark (with topology)...")
        ollama_results = benchmark_ollama_models(category, problems)
        
        print(f"  Running baseline models benchmark (no topology)...")
        baseline_results = benchmark_baseline_models(category, problems)
        
        # Initialize model entries if they don't exist
        for model in OLLAMA_MODELS:
            if model not in results['ollama']:
                results['ollama'][model] = []
            if model not in results['baseline']:
                results['baseline'][model] = []
                
            # Extend with this category's results
            if model in ollama_results:
                results['ollama'][model].extend(ollama_results[model])
            if model in baseline_results:
                results['baseline'][model].extend(baseline_results[model])
    
    return results

def analyze_topology_by_category(benchmark_results):
    """Analyze topology selection patterns by problem category."""
    solar_results = benchmark_results['solar']
    
    # Group by category
    categories = {}
    for result in solar_results:
        category = result['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    # Calculate topology distribution per category
    topology_distribution = {}
    for category, results in categories.items():
        topology_counts = {}
        for result in results:
            topo = result['selected_topology']
            topology_counts[topo] = topology_counts.get(topo, 0) + 1
        
        # Calculate percentages
        total = len(results)
        topology_distribution[category] = {
            topo: count/total for topo, count in topology_counts.items()
        }
    
    return topology_distribution

def analyze_performance_by_category(benchmark_results):
    """Analyze performance metrics by problem category."""
    performance = {
        'latency': {
            'solar': {},
            'hybrid': {}
        }
    }
    
    # For each Ollama model with topology
    for model in benchmark_results['ollama']:
        performance['latency'][model] = {}
    
    # For each baseline model (without topology)
    if 'baseline' in benchmark_results:
        for model in benchmark_results['baseline']:
            performance['latency'][f'baseline_{model}'] = {}
    
    # Calculate average latency by category for SOLAR
    for result in benchmark_results['solar']:
        category = result['category']
        if category not in performance['latency']['solar']:
            performance['latency']['solar'][category] = []
        performance['latency']['solar'][category].append(result['avg_latency'])
    
    # Calculate average latency by category for Hybrid
    for result in benchmark_results['hybrid']:
        category = result['category']
        if category not in performance['latency']['hybrid']:
            performance['latency']['hybrid'][category] = []
        performance['latency']['hybrid'][category].append(result['avg_latency'])
    
    # Calculate average latency by category for each Ollama model with topology
    for model, results in benchmark_results['ollama'].items():
        for result in results:
            category = result['category']
            if category not in performance['latency'][model]:
                performance['latency'][model][category] = []
            performance['latency'][model][category].append(result['avg_latency'])
    
    # Calculate average latency by category for each baseline model
    if 'baseline' in benchmark_results:
        for model, results in benchmark_results['baseline'].items():
            model_key = f'baseline_{model}'
            for result in results:
                category = result['category']
                if category not in performance['latency'][model_key]:
                    performance['latency'][model_key][category] = []
                performance['latency'][model_key][category].append(result['avg_latency'])
    
    # Convert lists to averages
    for approach in performance['latency']:
        for category in performance['latency'][approach]:
            latencies = performance['latency'][approach][category]
            performance['latency'][approach][category] = statistics.mean(latencies) if latencies else 0
    
    return performance

if __name__ == "__main__":
    print("Starting comprehensive SOLAR Framework benchmarking...")
    
    # Run all benchmarks
    benchmark_results = run_benchmarks()
    
    # Analyze topology selection patterns
    topology_analysis = analyze_topology_by_category(benchmark_results)
    benchmark_results['analysis'] = {
        'topology_by_category': topology_analysis,
        'performance_by_category': analyze_performance_by_category(benchmark_results)
    }
    
    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'solar_comprehensive_benchmark_{timestamp}.json'
    
    # Save results to file
    with open(filename, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\nBenchmark complete! Results saved to {filename}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nTopology Selection by Problem Category:")
    for category, distribution in topology_analysis.items():
        print(f"  {category}:")
        for topo, percentage in distribution.items():
            print(f"    {topo}: {percentage*100:.1f}%")
    
    print("\nAverage Latency by Problem Category (ms):")
    for approach, categories in benchmark_results['analysis']['performance_by_category']['latency'].items():
        print(f"  {approach}:")
        for category, avg_latency in categories.items():
            # Convert to milliseconds for display
            print(f"    {category}: {avg_latency*1000:.2f} ms")