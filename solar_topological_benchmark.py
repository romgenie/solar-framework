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
    
    # Measure total processing time including generation, reasoning, and evaluation
    start_time = time.time()
    
    # Generate the response using the specified topology
    response = topology.generate_response(problem_statement)
    
    # Track generation completion time
    generation_end_time = time.time()
    generation_latency = generation_end_time - start_time
    
    # Perform topological reasoning analysis - measure how well the response follows the topology structure
    topology_score = 0.0
    if isinstance(topology, ChainOfThought):
        # For CoT, check for sequential reasoning steps
        step_markers = sum(1 for i in range(1, 5) if f"Step {i}" in response)
        topology_score = min(step_markers / 3.0, 1.0)  # Normalize to [0,1]
        
    elif isinstance(topology, TreeOfThought):
        # For ToT, check for branch structure and conclusion
        has_branches = all(marker in response for marker in ["Branch A", "Branch B"]) 
        has_conclusion = "Conclusion" in response
        topology_score = (0.7 if has_branches else 0.0) + (0.3 if has_conclusion else 0.0)
        
    elif isinstance(topology, GraphOfThought):
        # For GoT, check for node structure and connections
        node_count = sum(1 for i in range(1, 5) if f"Node {i}" in response)
        has_connections = "Connection" in response or "Relationships" in response
        topology_score = min(node_count / 3.0, 0.7) + (0.3 if has_connections else 0.0)
    
    # Calculate reward score based on ground truth and topology adherence
    if ground_truth:
        # Use the reward model's scoring function but apply our topology weight
        base_reward = reward_model.score_response(response, ground_truth)
        reward_score = 0.6 * base_reward + 0.4 * topology_score
    else:
        # Without ground truth, give more weight to the topology structure
        response_quality = min(len(response) / 500, 1.0)  # Crude quality measure based on length
        reward_score = 0.3 * response_quality + 0.7 * topology_score
    
    # Extract the answer from the response
    extracted_answer = reward_model.extract_answer(response) if response else None
    
    # Calculate accuracy if ground truth is available
    accuracy = 0.0
    if ground_truth and extracted_answer:
        # Compute string similarity for numerical answer
        base_similarity = reward_model.compute_string_similarity(extracted_answer, ground_truth)
        
        # Apply problem-specific enhancements
        if "math" in problem_statement.lower() or "algebra" in problem_statement.lower():
            # For math problems, try to extract numerical answers for exact match
            import re
            extracted_numbers = re.findall(r'\d+(?:\.\d+)?', extracted_answer)
            ground_truth_numbers = re.findall(r'\d+(?:\.\d+)?', ground_truth)
            
            # If we have extracted numbers, check for exact matches
            if extracted_numbers and ground_truth_numbers:
                number_match = any(en == gtn for en in extracted_numbers for gtn in ground_truth_numbers)
                if number_match:
                    base_similarity = max(base_similarity, 0.9)  # Boost score for exact number match
        
        accuracy = base_similarity
    
    end_time = time.time()
    total_latency = end_time - start_time
    
    return {
        'response': response,
        'latency': total_latency,  # Use real total processing time
        'generation_latency': generation_latency,  # Just the response generation time
        'topology_score': topology_score,  # How well the response follows the topology
        'reward_score': reward_score,
        'extracted_answer': extracted_answer,
        'accuracy': accuracy
    }

def evaluate_ollama_with_topology_prompt(problem_statement, topology_name, ground_truth=None):
    """
    Evaluate Ollama model with a topology-specific prompt template.
    
    Parameters:
        problem_statement: The problem to solve.
        topology_name: The topology to simulate ("cot", "tot", or "got").
        ground_truth: Optional ground truth answer for verification.
        
    Returns:
        dict: Results including response, latency, reward scores, and accuracy.
    """
    baseline_model = BaselineModel(model_name=OLLAMA_MODEL)
    reward_model = MultiTaskTopologicalRewardModel()
    
    # Start measuring total processing time
    start_time = time.time()
    
    # Create topology-specific prompt with enhanced instructions
    if topology_name == "cot":
        prompt = f"""Please solve the following problem using chain-of-thought reasoning. 
Break down your solution into step-by-step sequential reasoning, clearly numbering each step.
Make sure to think through the problem thoroughly before concluding.

PROBLEM:
{problem_statement}

SOLUTION:
Step 1: """
    elif topology_name == "tot":
        prompt = f"""Please solve the following problem using tree-of-thought reasoning. 
Consider multiple possible approaches and reason through each branch before arriving at your conclusion.
Explore different solution paths and clearly label each branch of your thinking.

PROBLEM:
{problem_statement}

SOLUTION:
Branch A (First approach):
[Reasoning for first approach]

Branch B (Alternative approach):
[Reasoning for second approach]

Conclusion:
[Final answer with justification]"""
    elif topology_name == "got":
        prompt = f"""Please solve the following problem using graph-of-thought reasoning.
Consider multiple interconnected concepts and their relationships. Identify key nodes in your thinking process
and explicitly describe how these concepts connect to each other.

PROBLEM:
{problem_statement}

SOLUTION:
Node 1 (Initial understanding):
[Describe initial approach or concept]

Node 2 (Key insight):
[Describe an important insight or concept]

Node 3 (Alternative perspective):
[Describe another relevant concept or approach]

Relationships:
[Describe how the nodes connect or influence each other]

Final answer:
[Provide your conclusion with justification]"""
    else:
        prompt = SimplePromptTemplate.format_problem(problem_statement)
    
    # Process request with the external LLM
    process_start_time = time.time()
    result = baseline_model.process_request(prompt)
    process_end_time = time.time()
    process_latency = process_end_time - process_start_time
    
    # If successful, perform topology analysis and calculate reward score
    if result['success']:
        response = result['response']
        
        # Measure topology adherence
        topology_score = 0.0
        if topology_name == "cot":
            # For CoT, check for sequential step structure
            step_markers = sum(1 for i in range(1, 5) if f"Step {i}" in response)
            logical_flow = any(marker in response.lower() for marker in ["therefore", "thus", "so", "hence"])
            topology_score = min(step_markers / 3.0, 0.7) + (0.3 if logical_flow else 0.0)
            
        elif topology_name == "tot":
            # For ToT, check for branch structure and conclusion
            has_branch_a = "Branch A" in response or "First approach" in response
            has_branch_b = "Branch B" in response or "Alternative approach" in response
            has_conclusion = "Conclusion" in response or "Final answer" in response
            topology_score = (0.4 if has_branch_a else 0.0) + (0.3 if has_branch_b else 0.0) + (0.3 if has_conclusion else 0.0)
            
        elif topology_name == "got":
            # For GoT, check for node structure and relationships
            node_markers = ["Node 1", "Node 2", "Node 3", "Initial", "Key insight", "perspective"]
            node_count = sum(1 for marker in node_markers if marker in response)
            has_relationships = "Relationship" in response or "connect" in response.lower()
            topology_score = min(node_count / 4.0, 0.7) + (0.3 if has_relationships else 0.0)
        
        # Calculate content quality score and extract answer
        extracted_answer = reward_model.extract_answer(response)
        
        # Calculate similarity to ground truth if available
        content_score = 0.0
        accuracy = 0.0
        
        if ground_truth and extracted_answer:
            # Calculate similarity
            base_similarity = reward_model.compute_string_similarity(extracted_answer, ground_truth)
            
            # Apply problem-specific enhancements
            if "math" in problem_statement.lower() or "algebra" in problem_statement.lower():
                # For math problems, try to extract numerical answers for exact match
                import re
                extracted_numbers = re.findall(r'\d+(?:\.\d+)?', extracted_answer)
                ground_truth_numbers = re.findall(r'\d+(?:\.\d+)?', ground_truth)
                
                # If we have extracted numbers, check for exact matches
                if extracted_numbers and ground_truth_numbers:
                    number_match = any(en == gtn for en in extracted_numbers for gtn in ground_truth_numbers)
                    if number_match:
                        base_similarity = max(base_similarity, 0.9)  # Boost score for exact number match
            
            content_score = base_similarity
            accuracy = base_similarity
        else:
            # If no ground truth, estimate content quality based on detail and coherence
            content_score = min(len(response) / 800, 0.8)  # Crude quality measure based on length
        
        # Calculate final reward score - weight towards topology adherence for LLM+X approaches
        reward_score = 0.4 * topology_score + 0.6 * content_score
        
        # Add calculated metrics to the result
        result['reward_score'] = reward_score
        result['topology_score'] = topology_score
        result['content_score'] = content_score
        result['extracted_answer'] = extracted_answer
        result['accuracy'] = accuracy
    else:
        # If request failed, set all scores to 0
        result['reward_score'] = 0
        result['topology_score'] = 0
        result['content_score'] = 0
        result['extracted_answer'] = None
        result['accuracy'] = 0.0
    
    # Calculate total evaluation time
    end_time = time.time()
    total_latency = end_time - start_time
    
    # Add latency measurements
    result['total_latency'] = total_latency
    result['process_latency'] = process_latency
    
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
                
                # Pass ground truth to the evaluation function
                result = evaluate_ollama_with_topology_prompt(problem, prompt_type, ground_truth)
                
                if result['success']:
                    # Use total_latency for true end-to-end time
                    if 'total_latency' in result:
                        run_times.append(result['total_latency'])
                    else:
                        run_times.append(result['latency'])
                        
                    run_scores.append(result['reward_score'])
                    run_responses.append(result['response'])
                    
                    # Use pre-computed accuracy if available
                    if 'accuracy' in result and result['accuracy'] > 0:
                        run_accuracies.append(result['accuracy'])
                        
                    # Use pre-extracted answer if available
                    if 'extracted_answer' in result and result['extracted_answer']:
                        extracted_answers.append(result['extracted_answer'])
        
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
            
            # Calculate success rate (percent of runs with good answers)
            success_rate = 0.0
            if run_accuracies:
                success_rate = sum(1 for acc in run_accuracies if acc > 0.5) / len(run_accuracies)
                
            # Track topology adherence scores if available
            topology_scores = []
            if isinstance(result, dict) and 'topology_score' in result:
                topology_scores.append(result['topology_score'])
            
            # Calculate win metrics - helps determine which topology performed best
            win_metric = avg_accuracy * 0.7 + avg_score * 0.3  # Weight accuracy more heavily
                
            results.append({
                'problem': problem,
                'category': problem_category,
                'topology': topology_name,
                'ground_truth': ground_truth,
                'extracted_answer': most_common_answer,
                'accuracy': avg_accuracy,
                'avg_latency': avg_time,
                'avg_reward_score': avg_score,
                'success_rate': success_rate,
                'win_metric': win_metric,
                'topology_adherence': statistics.mean(topology_scores) if topology_scores else 0.0,
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
    
    # === Create win metric by category visualization ===
    if 'win_metric' in df.columns:
        best_win_metric = df.groupby(['category', 'topology'])['win_metric'].mean().reset_index()
        best_win_metric = best_win_metric.sort_values(['category', 'win_metric'], ascending=[True, False])
        best_by_win_metric = best_win_metric.groupby('category').first().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='category', y='win_metric', data=best_by_win_metric, 
                   hue='topology', palette='plasma')
        plt.title('Overall Best Performing Topology by Problem Category', fontsize=16)
        plt.xlabel('Problem Category', fontsize=14)
        plt.ylabel('Combined Performance Metric', fontsize=14)
        plt.legend(title='Topology')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/best_topology_combined_metric.png')
        plt.close()
        
        # Add to report
        report['best_topology_by_win_metric'] = best_by_win_metric.to_dict(orient='records')
        report['average_win_metric'] = df.groupby('topology')['win_metric'].mean().to_dict()

    # === Create success rate visualization ===
    if 'success_rate' in df.columns:
        plt.figure(figsize=(14, 8))
        sns.barplot(x='category', y='success_rate', hue='topology', data=df)
        plt.title('Success Rate by Category and Topology', fontsize=16)
        plt.xlabel('Problem Category', fontsize=14)
        plt.ylabel('Success Rate', fontsize=14)
        plt.legend(title='Topology')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/success_rate_by_category.png')
        plt.close()
        
        # Add to report
        report['average_success_rate'] = df.groupby('topology')['success_rate'].mean().to_dict()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    
    # Print win metric statistics if available
    if 'win_metric' in df.columns:
        print("\nBest Overall Topology by Category (Combined Metric):")
        for item in best_by_win_metric.to_dict(orient='records'):
            print(f"  {item['category']}: {item['topology']} (Score: {item['win_metric']:.4f})")
        
        print("\nAverage Combined Performance Metric:")
        for topo, score in sorted(report['average_win_metric'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {topo}: {score:.4f}")
    
    # Print accuracy statistics if available
    if 'accuracy' in df.columns:
        print("\nMost Accurate Topology by Category:")
        for item in best_by_category_accuracy.to_dict(orient='records'):
            print(f"  {item['category']}: {item['topology']} (Accuracy: {item['accuracy']:.4f})")
        
        print("\nAverage Accuracy:")
        for topo, acc in sorted(report['average_accuracy'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {topo}: {acc:.4f}")
    
    # Print success rate if available
    if 'success_rate' in df.columns:
        print("\nAverage Success Rate:")
        for topo, rate in sorted(report['average_success_rate'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {topo}: {rate:.4f}")
    
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