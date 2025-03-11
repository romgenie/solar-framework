"""
Baseline Model implementation for SOLAR framework comparison.

This module implements a simple baseline LLM approach without topology selection,
to serve as a control for the SOLAR framework evaluation.
"""

import random
import time
import requests

class BaselineModel:
    """
    Baseline model that directly passes prompts to an LLM without any
    topology-aware routing or reward-based selection.
    
    This serves as a control to evaluate the impact of the SOLAR framework's
    topology selection and rewarding mechanisms.
    """
    
    def __init__(self, model_name="llama3.2"):
        """
        Initialize the baseline model.
        
        Parameters:
            model_name (str): The name of the LLM to use via Ollama.
        """
        self.model_name = model_name
    
    def process_request(self, problem_statement):
        """
        Process a problem statement by directly sending it to the LLM.
        
        Parameters:
            problem_statement (str): The problem statement to solve.
            
        Returns:
            dict: Contains:
                - response (str): The LLM's response.
                - latency (float): The time taken to generate the response.
                - success (bool): Whether the request was successful.
        """
        start_time = time.time()
        
        try:
            # Call the Ollama API
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': problem_statement,
                    'stream': False
                },
                timeout=60
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'response': result['response'],
                    'latency': end_time - start_time,
                    'success': True
                }
            else:
                return {
                    'response': f"Error: HTTP {response.status_code}",
                    'latency': end_time - start_time,
                    'success': False
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                'response': f"Error: {str(e)}",
                'latency': end_time - start_time,
                'success': False
            }


class SimplePromptTemplate:
    """
    A simple prompt template generator to ensure consistent formatting
    across baseline and SOLAR models.
    """
    
    @staticmethod
    def format_problem(problem_statement):
        """
        Format a problem statement for the model.
        
        Parameters:
            problem_statement (str): The problem statement to format.
            
        Returns:
            str: The formatted prompt.
        """
        return f"""Please solve the following problem step by step:

{problem_statement}

Show your reasoning and final answer.
"""


if __name__ == "__main__":
    # Example usage
    baseline = BaselineModel()
    
    problem = "Solve the math problem: What is 2+2?"
    prompt = SimplePromptTemplate.format_problem(problem)
    
    result = baseline.process_request(prompt)
    
    print(f"Problem: {problem}")
    print(f"Latency: {result['latency']:.4f} seconds")
    print(f"Response: {result['response']}")