"""
SOLAR Topological Components Benchmark

This script specifically benchmarks the individual reasoning topologies (Chain of Thought, 
Tree of Thought, Graph of Thought) against each other across different problem categories.
This helps understand which topology performs best for which problem types.

Uses standard benchmark datasets:
- GSM8K (Grade School Math)
- AQUA-RAT (Algebra Question Answering with Rationales)
- LogiQA (Logical Reasoning)
- CRT (Cognitive Reflection Test)
"""

import time
import json
import statistics
import random
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from solar_config import config
from solar_topological_rewarding import ChainOfThought, TreeOfThought, GraphOfThought, MultiTaskTopologicalRewardModel
from baseline_model import SimplePromptTemplate, BaselineModel
from solar_hybrid_scaling import FineTunedModel
from solar_dataset_loader import DatasetLoader

# Test parameters - configurable from config.ini
NUM_RUNS = config.get_num_runs()
PROBLEMS_PER_CATEGORY = config.get_problems_per_category()
OLLAMA_MODEL = config.get_ollama_model()  # Model to use for LLM generation

# Load benchmark problems from standard datasets
loader = DatasetLoader()
TEST_PROBLEMS = loader.load_benchmark_problems(sample_size=PROBLEMS_PER_CATEGORY)

# Fallback test problems (used if dataset loading fails)
FALLBACK_PROBLEMS = {
    "arithmetic": [
        "Solve the math problem: What is 25 + 18?",
    ],
    "algebra": [
        "Solve the math problem: If 3x + 7 = 22, what is x?",
    ],
    "logic": [
        "Solve this logic problem: All roses are flowers. Some flowers fade quickly. Therefore, some roses fade quickly. Is this conclusion valid?",
    ],
    "edge_cases": [
        "Solve this ambiguous problem: If a bat and a ball cost $1.10 in total, and the bat costs $1.00 more than the ball, how much does the ball cost?",
    ]
}

# Use fallback problems for any category with empty problems
for category in TEST_PROBLEMS:
    if not TEST_PROBLEMS[category]:
        print(f"Warning: No problems loaded for {category}, using fallback problems")
        TEST_PROBLEMS[category] = FALLBACK_PROBLEMS[category]

def evaluate_topology(topology_class, problem_statement, ground_truth=None):
    """
    Evaluate a specific topology on a problem statement.
    
    Parameters:
        topology_class: The class of the topology to evaluate.
        problem_statement: The problem to solve.
        ground_truth: Optional ground truth answer for verification.
        
    Returns:
        dict: Results including response, latency, reward score, and accuracy.
    """
    topology = topology_class()
    reward_model = MultiTaskTopologicalRewardModel()
    
    # Measure generation time
    start_time = time.time()
    response = topology.generate_response(problem_statement)
    end_time = time.time()
    latency = end_time - start_time
    
    # Calculate reward score based on ground truth if available
    reward_score = reward_model.score_response(response, ground_truth)
    
    # Extract the answer from the response
    extracted_answer = reward_model.extract_answer(response) if ground_truth else None
    
    # Calculate accuracy if ground truth is available
    accuracy = 0.0
    if ground_truth and extracted_answer:
        accuracy = reward_model.compute_string_similarity(extracted_answer, ground_truth)
    
    return {
        'response': response,
        'latency': latency,
        'reward_score': reward_score,
        'extracted_answer': extracted_answer,
        'accuracy': accuracy
    }

def evaluate_ollama_with_topology_prompt(problem_statement, topology_name):
    """
    Evaluate Ollama model with a topology-specific prompt template.
    
    Parameters:
        problem_statement: The problem to solve.
        topology_name: The topology to simulate ("cot", "tot", or "got").
        
    Returns:
        dict: Results including response, latency, and success status.
    """
    baseline_model = BaselineModel(model_name=OLLAMA_MODEL)
    
    # Create topology-specific prompt
    if topology_name == "cot":
        prompt = f"""Please solve the following problem using chain-of-thought reasoning. 
Break down your solution into step-by-step sequential reasoning:

{problem_statement}

Think step by step:
1. """
    elif topology_name == "tot":
        prompt = f"""Please solve the following problem using tree-of-thought reasoning. 
Consider multiple possible approaches and reason through each branch before arriving at your conclusion:

{problem_statement}

Approach A:
Approach B:
Conclusion:"""
    elif topology_name == "got":
        prompt = f"""Please solve the following problem using graph-of-thought reasoning.
Consider multiple interconnected concepts and their relationships:

{problem_statement}

Node 1 (Initial concept):
Node 2 (Related concept):
Node 3 (Another perspective):
Relationships:
Final answer:"""
    else:
        prompt = SimplePromptTemplate.format_problem(problem_statement)
    
    # Process request
    start_time = time.time()
    result = baseline_model.process_request(prompt)
    end_time = time.time()
    
    # If successful, calculate reward score
    if result['success']:
        reward_model = MultiTaskTopologicalRewardModel()
        reward_score = reward_model.score_response(result['response'])
        result['reward_score'] = reward_score
    else:
        result['reward_score'] = 0
    
    return result

def benchmark_topology(topology_name, problem_category, problems):
    """
    Benchmark a specific topology on a set of problems.
    
    Parameters:
        topology_name: The name of the topology to benchmark.
        problem_category: The category of problems being evaluated.
        problems: List of problem statements.
        
    Returns:
        list: Results for each problem.
    """
    results = []
    
    # Load ground truth answers for each problem
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
        run_scores = []
        run_responses = []
        run_accuracies = []
        extracted_answers = []
        
        # Find the ground truth for this problem
        ground_truth = None
        if problem in problem_details:
            if 'answer' in problem_details[problem]:
                ground_truth = problem_details[problem]['answer']
            elif 'correct_answer' in problem_details[problem]:
                ground_truth = problem_details[problem]['correct_answer']
        
        for _ in range(NUM_RUNS):
            # Determine which topology to use
            if topology_name == "ChainOfThought":
                result = evaluate_topology(ChainOfThought, problem, ground_truth)
                run_times.append(result['latency'])
                run_scores.append(result['reward_score'])
                run_responses.append(result['response'])
                if 'accuracy' in result:
                    run_accuracies.append(result['accuracy'])
                if 'extracted_answer' in result and result['extracted_answer']:
                    extracted_answers.append(result['extracted_answer'])
            elif topology_name == "TreeOfThought":
                result = evaluate_topology(TreeOfThought, problem, ground_truth)
                run_times.append(result['latency'])
                run_scores.append(result['reward_score'])
                run_responses.append(result['response'])
                if 'accuracy' in result:
                    run_accuracies.append(result['accuracy'])
                if 'extracted_answer' in result and result['extracted_answer']:
                    extracted_answers.append(result['extracted_answer'])
            elif topology_name == "GraphOfThought":
                result = evaluate_topology(GraphOfThought, problem, ground_truth)
                run_times.append(result['latency'])
                run_scores.append(result['reward_score'])
                run_responses.append(result['response'])
                if 'accuracy' in result:
                    run_accuracies.append(result['accuracy'])
                if 'extracted_answer' in result and result['extracted_answer']:
                    extracted_answers.append(result['extracted_answer'])
            elif topology_name in ["LLM+CoT", "LLM+ToT", "LLM+GoT"]:
                # Map to the appropriate topology prompt type
                if topology_name == "LLM+CoT":
                    prompt_type = "cot"
                elif topology_name == "LLM+ToT":
                    prompt_type = "tot"
                else:
                    prompt_type = "got"
                
                result = evaluate_ollama_with_topology_prompt(problem, prompt_type)
                if result['success']:
                    run_times.append(result['latency'])
                    run_scores.append(result['reward_score'])
                    run_responses.append(result['response'])
                    
                    # Extract answer and calculate accuracy if ground truth available
                    if ground_truth:
                        reward_model = MultiTaskTopologicalRewardModel()
                        extracted_answer = reward_model.extract_answer(result['response'])
                        extracted_answers.append(extracted_answer)
                        accuracy = reward_model.compute_string_similarity(extracted_answer, ground_truth)
                        run_accuracies.append(accuracy)
        
        # Calculate statistics
        if run_times:
            avg_time = statistics.mean(run_times)
            avg_score = statistics.mean(run_scores)
            avg_accuracy = statistics.mean(run_accuracies) if run_accuracies else 0.0
            
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
                'topology': topology_name,
                'ground_truth': ground_truth,
                'extracted_answer': most_common_answer,
                'accuracy': avg_accuracy,
                'avg_latency': avg_time,
                'avg_reward_score': avg_score,
                'response': run_responses[0] if run_responses else "No response",
            })
    
    return results

def run_topology_benchmarks():
    """Run benchmarks for all topologies across all problem categories."""
    topologies = [
        "ChainOfThought", 
        "TreeOfThought", 
        "GraphOfThought",
        "LLM+CoT",
        "LLM+ToT",
        "LLM+GoT"
    ]
    
    results = {
        'topologies': [],
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_runs': NUM_RUNS,
            'ollama_model': OLLAMA_MODEL
        }
    }
    
    # Run benchmarks for each topology and problem category
    for topology in topologies:
        print(f"\nBenchmarking {topology}...")
        
        for category, problems in TEST_PROBLEMS.items():
            print(f"  Testing on {category} problems...")
            category_results = benchmark_topology(topology, category, problems)
            results['topologies'].extend(category_results)
    
    return results

def analyze_topology_results(benchmark_results):
    """Analyze the topology benchmark results."""
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(benchmark_results['topologies'])
    
    # Create output directory
    import os
    output_dir = 'solar_topology_benchmark'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # === Analyze reward scores by topology and category ===
    plt.figure(figsize=(14, 8))
    sns.barplot(x='category', y='avg_reward_score', hue='topology', data=df)
    plt.title('Average Reward Score by Category and Topology', fontsize=16)
    plt.xlabel('Problem Category', fontsize=14)
    plt.ylabel('Average Reward Score', fontsize=14)
    plt.legend(title='Topology')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reward_scores_by_category.png')
    plt.close()
    
    # === Analyze accuracy by topology and category ===
    if 'accuracy' in df.columns:
        plt.figure(figsize=(14, 8))
        sns.barplot(x='category', y='accuracy', hue='topology', data=df)
        plt.title('Average Accuracy by Category and Topology', fontsize=16)
        plt.xlabel('Problem Category', fontsize=14)
        plt.ylabel('Accuracy (similarity to ground truth)', fontsize=14)
        plt.legend(title='Topology')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/accuracy_by_category.png')
        plt.close()
    
    # === Analyze latency by topology and category ===
    plt.figure(figsize=(14, 8))
    sns.barplot(x='category', y='avg_latency', hue='topology', data=df)
    plt.title('Average Latency by Category and Topology', fontsize=16)
    plt.xlabel('Problem Category', fontsize=14)
    plt.ylabel('Average Latency (seconds)', fontsize=14)
    plt.legend(title='Topology')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latency_by_category.png')
    plt.close()
    
    # === Create a logarithmic scale version for latency ===
    plt.figure(figsize=(14, 8))
    sns.barplot(x='category', y='avg_latency', hue='topology', data=df)
    plt.title('Average Latency by Category and Topology (Log Scale)', fontsize=16)
    plt.xlabel('Problem Category', fontsize=14)
    plt.ylabel('Average Latency (seconds) - Log Scale', fontsize=14)
    plt.yscale('log')
    plt.legend(title='Topology')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latency_by_category_log.png')
    plt.close()
    
    # === Calculate the best topology for each category based on accuracy ===
    if 'accuracy' in df.columns:
        best_topology_accuracy = df.groupby(['category', 'topology'])['accuracy'].mean().reset_index()
        best_topology_accuracy = best_topology_accuracy.sort_values(['category', 'accuracy'], ascending=[True, False])
        best_by_category_accuracy = best_topology_accuracy.groupby('category').first().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='category', y='accuracy', data=best_by_category_accuracy, 
                   hue='topology', palette='viridis')
        plt.title('Most Accurate Topology by Problem Category', fontsize=16)
        plt.xlabel('Problem Category', fontsize=14)
        plt.ylabel('Accuracy (similarity to ground truth)', fontsize=14)
        plt.legend(title='Topology')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/best_topology_by_accuracy.png')
        plt.close()
    
    # === Calculate the best topology for each category based on reward score ===
    best_topology = df.groupby(['category', 'topology'])['avg_reward_score'].mean().reset_index()
    best_topology = best_topology.sort_values(['category', 'avg_reward_score'], ascending=[True, False])
    best_by_category = best_topology.groupby('category').first().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='category', y='avg_reward_score', data=best_by_category, 
               hue='topology', palette='viridis')
    plt.title('Best Performing Topology by Problem Category', fontsize=16)
    plt.xlabel('Problem Category', fontsize=14)
    plt.ylabel('Average Reward Score', fontsize=14)
    plt.legend(title='Topology')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_topology_by_reward.png')
    plt.close()
    
    # === Ground Truth Analysis (if available) ===
    if 'ground_truth' in df.columns and 'extracted_answer' in df.columns:
        # Create summary of answers vs ground truth
        answer_comparison = df[['problem', 'category', 'topology', 'ground_truth', 'extracted_answer', 'accuracy']]
        
        # Save the comparison to a CSV file
        answer_comparison.to_csv(f'{output_dir}/answer_comparison.csv', index=False)
        
        # Save a text report
        with open(f'{output_dir}/answer_verification.txt', 'w') as f:
            f.write("=== Ground Truth vs Extracted Answers ===\n\n")
            
            for category in sorted(df['category'].unique()):
                f.write(f"\n\n== CATEGORY: {category.upper()} ==\n\n")
                category_df = df[df['category'] == category]
                
                for _, row in category_df.iterrows():
                    f.write(f"Problem: {row['problem'][:100]}...\n")
                    f.write(f"Topology: {row['topology']}\n")
                    f.write(f"Ground Truth: {row['ground_truth']}\n")
                    f.write(f"Extracted Answer: {row['extracted_answer']}\n")
                    f.write(f"Accuracy: {row['accuracy']:.4f}\n")
                    f.write("-" * 80 + "\n")
    
    # === Generate report ===
    report = {
        'best_topology_by_category': best_by_category.to_dict(orient='records'),
        'average_scores': df.groupby('topology')['avg_reward_score'].mean().to_dict(),
        'average_latency': df.groupby('topology')['avg_latency'].mean().to_dict()
    }
    
    # Add accuracy metrics if available
    if 'accuracy' in df.columns:
        report['average_accuracy'] = df.groupby('topology')['accuracy'].mean().to_dict()
        report['best_topology_by_accuracy'] = best_by_category_accuracy.to_dict(orient='records') if 'best_by_category_accuracy' in locals() else {}
    
    # Save results and report
    with open(f'{output_dir}/topology_benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    with open(f'{output_dir}/topology_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    
    # Print accuracy statistics if available
    if 'accuracy' in df.columns:
        print("\nMost Accurate Topology by Category:")
        for item in best_by_category_accuracy.to_dict(orient='records'):
            print(f"  {item['category']}: {item['topology']} (Accuracy: {item['accuracy']:.4f})")
        
        print("\nAverage Accuracy:")
        for topo, acc in sorted(report['average_accuracy'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {topo}: {acc:.4f}")
    
    print("\nBest Scoring Topology by Category:")
    for item in best_by_category.to_dict(orient='records'):
        print(f"  {item['category']}: {item['topology']} (Score: {item['avg_reward_score']:.4f})")
    
    print("\nAverage Reward Scores:")
    for topo, score in sorted(report['average_scores'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {topo}: {score:.4f}")
    
    print("\nAverage Latency (seconds):")
    for topo, latency in sorted(report['average_latency'].items(), key=lambda x: x[1]):
        print(f"  {topo}: {latency:.6f}")

if __name__ == "__main__":
    print("Starting SOLAR Topological Components Benchmark...")
    
    # Run topology benchmarks
    benchmark_results = run_topology_benchmarks()
    
    # Analyze results
    analyze_topology_results(benchmark_results)