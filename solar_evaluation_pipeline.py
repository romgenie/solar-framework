"""
This module implements the Evaluation Pipeline for the SOLAR framework.
It simulates evaluation of multiple reasoning topologies (Chain-of-Thought, Tree-of-Thought, Graph-of-Thought)
using metrics such as Accuracy and Win Rate.

The evaluation process involves:
1. Taking a set of problem statements.
2. Generating responses using different reasoning topologies via the TAG System.
3. Annotating each response with simulated Topo Labels (continuous scores) and Hard Labels (binary correctness).
4. Computing Accuracy for each topology as the proportion of correct responses.
5. Computing Win Rate for each topology as the fraction of questions where the topology had the highest Topo Label.
"""

import random

class EvaluationPipeline:
    """
    EvaluationPipeline simulates the evaluation of reasoning topologies
    by computing performance metrics across multiple problem statements.
    """
    def __init__(self, tag_system):
        """
        Initialize the EvaluationPipeline with an instance of TAGSystem.
        
        Parameters:
            tag_system (TAGSystem): An instance responsible for generating and annotating responses.
        """
        self.tag_system = tag_system
        self.topologies = self.tag_system.topologies
    
    def evaluate(self, problem_statements):
        """
        Evaluate a list of problem statements and compute accuracy and win rate metrics.
        
        Parameters:
            problem_statements (list of str): List of problem statements to evaluate.
        
        Returns:
            dict: A dictionary mapping each topology to its computed metrics:
                  {
                    'Chain-of-Thought': {'accuracy': <float>, 'win_rate': <float>},
                    'Tree-of-Thought': {'accuracy': <float>, 'win_rate': <float>},
                    'Graph-of-Thought': {'accuracy': <float>, 'win_rate': <float>}
                  }
        """
        total_questions = len(problem_statements)
        # Counters for accuracy and win rate calculations
        correct_counts = {topo: 0 for topo in self.topologies}
        win_counts = {topo: 0 for topo in self.topologies}
        total_counts = {topo: 0 for topo in self.topologies}
        
        for problem in problem_statements:
            # Generate responses and annotations using TAGSystem
            responses = self.tag_system.generate_responses(problem)
            annotations = self.tag_system.annotate_responses(responses)
            
            # For accuracy: count each response as correct if its hard label is 1.
            for topo in self.topologies:
                total_counts[topo] += 1
                # Hard label is 1 if response is correct
                if annotations[topo][1] == 1:
                    correct_counts[topo] += 1
            
            # For win rate: the topology with the highest topo label wins for this problem.
            highest_topo = max(annotations, key=lambda topo: annotations[topo][0])
            win_counts[highest_topo] += 1
        
        # Calculate accuracy and win rate for each topology.
        metrics = {}
        for topo in self.topologies:
            accuracy = correct_counts[topo] / total_counts[topo] if total_counts[topo] > 0 else 0
            win_rate = win_counts[topo] / total_questions if total_questions > 0 else 0
            metrics[topo] = {'accuracy': accuracy, 'win_rate': win_rate}
        
        return metrics

if __name__ == "__main__":
    # Import the TAGSystem from the solar_tag_system module
    from solar_tag_system import TAGSystem
    
    # Instantiate the TAGSystem which handles response generation and annotation.
    tag_system = TAGSystem()
    evaluation_pipeline = EvaluationPipeline(tag_system)
    
    # Define a list of problem statements for evaluation.
    problem_statements = [
        "Solve the math problem: What is 2 + 2?",
        "Solve the math problem: What is 5 * 3?",
        "Solve the math problem: What is 10 - 4?",
        "Solve the math problem: What is 8 / 2?",
        "Solve the math problem: What is 7 + 3?"
    ]
    
    # Run the evaluation process.
    metrics = evaluation_pipeline.evaluate(problem_statements)
    print("Evaluation Metrics:")
    for topo, data in metrics.items():
        print(f"{topo}: Accuracy = {data['accuracy']:.2f}, Win Rate = {data['win_rate']:.2f}")