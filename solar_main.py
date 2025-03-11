"""
This module serves as the main entry point to demonstrate the SOLAR framework's components
in a unified simulation. It integrates and exercises the following components:

1. Topological Rewarding (Inference Scaling) - using InferencePipeline from solar_topological_rewarding.
2. Hybrid Scaling (Combined Training and Inference Scaling) - using HybridScalingInference from solar_hybrid_scaling.
3. Topological-Annotation-Generation (TAG) System & Topological Tuning - using TAGSystem from solar_tag_system.
4. Evaluation Pipeline - using EvaluationPipeline from solar_evaluation_pipeline.

The main function runs through these components by:
- Evaluating multiple problem statements.
- Processing a sample problem via topological rewarding.
- Processing a sample problem via hybrid scaling.
- Demonstrating the TAG system's data generation, annotation, and fine-tuning simulation.
"""

from solar_topological_rewarding import InferencePipeline
from solar_hybrid_scaling import HybridScalingInference
from solar_tag_system import TAGSystem
from solar_evaluation_pipeline import EvaluationPipeline

def run_evaluation():
    """
    Run the evaluation pipeline on a set of sample problem statements.
    
    This function generates responses using the TAGSystem, annotates them,
    and computes simulated accuracy and win rate metrics across different reasoning topologies.
    
    Returns:
        dict: A dictionary of evaluation metrics for each reasoning topology.
    """
    tag_system = TAGSystem()
    evaluation_pipeline = EvaluationPipeline(tag_system)
    problem_statements = [
        "Solve the math problem: What is 2 + 2?",
        "Solve the math problem: What is 5 * 3?",
        "Solve the math problem: What is 10 - 4?",
        "Solve the math problem: What is 8 / 2?",
        "Solve the math problem: What is 7 + 3?"
    ]
    metrics = evaluation_pipeline.evaluate(problem_statements)
    return metrics

def run_topological_inference(problem_statement):
    """
    Process a problem statement using the Topological Rewarding (Inference Scaling) component.
    
    Parameters:
        problem_statement (str): The problem statement to solve.
        
    Returns:
        dict: Results containing the selected topology, response, reward score, and all generated responses.
    """
    pipeline = InferencePipeline()
    result = pipeline.process_request(problem_statement)
    return result

def run_hybrid_inference(problem_statement):
    """
    Process a problem statement using the Hybrid Scaling component.
    
    Parameters:
        problem_statement (str): The problem statement to solve.
        
    Returns:
        dict: Results containing the selected topology, fine-tuned response, reward score, and all generated responses.
    """
    hybrid_pipeline = HybridScalingInference()
    result = hybrid_pipeline.process_request(problem_statement)
    return result

def demonstrate_tag_system(problem_statement):
    """
    Demonstrate the Topological-Annotation-Generation (TAG) System functionality.
    
    This includes generating responses, annotating them, segmenting problem difficulty,
    preparing training data, and simulating fine-tuning.
    
    Parameters:
        problem_statement (str): The problem statement for which to generate data.
        
    Returns:
        tuple: Contains generated responses, annotations, problem difficulty, and training data.
    """
    tag_system = TAGSystem()
    responses = tag_system.generate_responses(problem_statement)
    annotations = tag_system.annotate_responses(responses)
    difficulty = tag_system.segment_problems(annotations)
    training_data = tag_system.prepare_training_data(problem_statement, responses, annotations)
    # Simulate fine-tuning (the function prints details during execution)
    fine_tuning_status = tag_system.fine_tune_model(training_data)
    return responses, annotations, difficulty, training_data, fine_tuning_status

def main():
    """
    Main function to run demonstrations of the SOLAR framework's components.
    """
    # Run and print evaluation metrics for a set of problem statements.
    eval_metrics = run_evaluation()
    print("=== Evaluation Metrics ===")
    for topo, metrics in eval_metrics.items():
        print(f"{topo}: Accuracy = {metrics['accuracy']:.2f}, Win Rate = {metrics['win_rate']:.2f}")
    
    # Demonstrate Topological Rewarding Inference
    sample_problem = "Solve the math problem: What is 3 + 7?"
    print("\n=== Topological Rewarding Inference ===")
    topological_result = run_topological_inference(sample_problem)
    print("Selected Topology:", topological_result['selected_topology'])
    print("Response:", topological_result['response'])
    print("Score:", topological_result['score'])
    print("All Responses:", topological_result['all_responses'])
    
    # Demonstrate Hybrid Scaling Inference
    sample_problem_hybrid = "Solve the math problem: What is 6 * 4?"
    print("\n=== Hybrid Scaling Inference ===")
    hybrid_result = run_hybrid_inference(sample_problem_hybrid)
    print("Selected Topology:", hybrid_result['selected_topology'])
    print("Response:", hybrid_result['response'])
    print("Score:", hybrid_result['score'])
    print("All Responses:", hybrid_result['all_responses'])
    
    # Demonstrate TAG System for data generation, annotation, and fine-tuning
    sample_problem_tag = "Solve the math problem: What is 9 - 3?"
    print("\n=== TAG System Demonstration ===")
    responses, annotations, difficulty, training_data, fine_tuning_status = demonstrate_tag_system(sample_problem_tag)
    print("Generated Responses:", responses)
    print("Annotations:", annotations)
    print("Problem Difficulty:", difficulty)
    print("Training Data:", training_data)
    print("Fine-Tuning Status:", fine_tuning_status)

if __name__ == "__main__":
    main()