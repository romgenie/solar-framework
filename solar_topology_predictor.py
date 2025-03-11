"""
SOLAR Topology Predictor Component

This module implements the core topology prediction functionality for the SOLAR framework.
It dynamically selects the most suitable reasoning topology for a given problem without
having to try all approaches first.

Components:
- ProblemFeatureExtractor: Extracts relevant features from problem statements
- TopologyPredictor: Predicts the most suitable topology based on problem features
- DynamicTopologyRouter: Routes problems to the optimal topology pipeline
"""

import re
import numpy as np
from collections import Counter

class ProblemFeatureExtractor:
    """
    Extracts features from problem statements to support topology prediction.
    
    These features include:
    - Problem domain (math, logic, etc.)
    - Complexity indicators (word count, special characters, etc.)
    - Problem structure indicators (presence of constraints, multiple steps, etc.)
    """
    
    def __init__(self):
        """Initialize feature extractors and classifiers."""
        # Domain keywords for categorization
        self.domain_keywords = {
            'math': ['add', 'subtract', 'multiply', 'divide', 'calculate', 'equation', 'sum', 'difference', 
                    'product', 'quotient', 'equals', 'equal to', 'solve for', 'value of', 'compute'],
            'algebra': ['solve', 'equation', 'variable', 'expression', 'simplify', 'factor', 'polynomial',
                       'linear', 'quadratic', 'system', 'unknown', 'find x', 'x =', 'y ='],
            'logic': ['valid', 'invalid', 'conclusion', 'premises', 'argument', 'logic', 'fallacy', 
                     'therefore', 'implies', 'deduction', 'induction', 'if-then', 'syllogism'],
            'complex': ['optimize', 'maximum', 'minimum', 'constraints', 'probability', 'likelihood',
                      'conditions', 'dependent', 'independent', 'factors', 'multiple', 'interconnected']
        }

    def extract_features(self, problem_statement):
        """
        Extract numerical and categorical features from a problem statement.
        
        Parameters:
            problem_statement (str): The problem to analyze
            
        Returns:
            dict: Features extracted from the problem
        """
        features = {}
        
        # Basic text statistics
        features['word_count'] = len(problem_statement.split())
        features['char_count'] = len(problem_statement)
        features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
        
        # Mathematical expressions and numbers
        features['contains_equations'] = 1 if re.search(r'[a-z]\s*=|=\s*[a-z]|equation', problem_statement.lower()) else 0
        features['number_of_numbers'] = len(re.findall(r'\b\d+\.?\d*\b', problem_statement))
        features['contains_math_operators'] = 1 if re.search(r'[\+\-\*\/\^]', problem_statement) else 0
        
        # Logic indicators
        features['contains_logic_keywords'] = 1 if re.search(r'\bif\b|\bthen\b|\bvalid\b|\bconclusion\b|\bargument\b', 
                                                           problem_statement.lower()) else 0
        features['contains_comparison'] = 1 if re.search(r'than|greater|less|equal|more|most|least|compare', 
                                                       problem_statement.lower()) else 0
        
        # Complexity indicators
        features['contains_multi_step'] = 1 if re.search(r'first|second|third|next|then|after|before|finally', 
                                                      problem_statement.lower()) else 0
        features['contains_constraints'] = 1 if re.search(r'constraint|condition|must|cannot|at least|at most|only if', 
                                                       problem_statement.lower()) else 0
        features['contains_uncertainty'] = 1 if re.search(r'probability|chance|likely|possibly|might|could', 
                                                       problem_statement.lower()) else 0
        
        # Domain detection
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            domain_scores[domain] = sum(1 for keyword in keywords if keyword.lower() in problem_statement.lower())
            features[f'{domain}_score'] = domain_scores[domain]
        
        # Primary domain
        features['primary_domain'] = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else 'unknown'
        
        # Cognitive reflection indicators (checking for misleading intuition)
        features['potential_cognitive_trap'] = 1 if re.search(r'bat and ball|machines|lily|widgets|hospital', 
                                                             problem_statement.lower()) else 0
        
        return features


class TopologyPredictor:
    """
    Predicts the most suitable reasoning topology based on problem features.
    
    Uses a rule-based system to determine whether Chain-of-Thought, Tree-of-Thought,
    or Graph-of-Thought would be most effective for a given problem.
    """
    
    def __init__(self):
        """Initialize the predictor with topology selection rules."""
        self.feature_extractor = ProblemFeatureExtractor()
        
    def predict_topology(self, problem_statement):
        """
        Predict the most suitable reasoning topology for the given problem.
        
        Parameters:
            problem_statement (str): The problem to analyze
            
        Returns:
            dict: Contains the predicted topology and confidence scores
        """
        # Extract features
        features = self.feature_extractor.extract_features(problem_statement)
        
        # Calculate suitability scores for each topology
        cot_score = self._calculate_cot_score(features)
        tot_score = self._calculate_tot_score(features)
        got_score = self._calculate_got_score(features)
        
        # Determine the best topology based on highest score
        scores = {
            'Chain-of-Thought': cot_score,
            'Tree-of-Thought': tot_score,
            'Graph-of-Thought': got_score
        }
        
        predicted_topology = max(scores, key=scores.get)
        confidence = scores[predicted_topology] / sum(scores.values()) if sum(scores.values()) > 0 else 0.33
        
        return {
            'predicted_topology': predicted_topology,
            'confidence': confidence,
            'topology_scores': scores,
            'features': features
        }
    
    def _calculate_cot_score(self, features):
        """Calculate suitability score for Chain-of-Thought reasoning."""
        score = 0.5  # Base score
        
        # CoT works well for straightforward sequential problems
        if features['primary_domain'] == 'math':
            score += 0.3
        
        if features['contains_multi_step'] == 1:
            score += 0.2
            
        # Lower score for problems with multiple possible approaches or constraints
        if features['contains_constraints'] == 1:
            score -= 0.1
            
        if features['potential_cognitive_trap'] == 1:
            score -= 0.3
            
        # Sequential calculations benefit from CoT
        if features['contains_math_operators'] == 1 and features['number_of_numbers'] > 1:
            score += 0.2
            
        return max(0.1, min(1.0, score))  # Ensure score is between 0.1 and 1.0
    
    def _calculate_tot_score(self, features):
        """Calculate suitability score for Tree-of-Thought reasoning."""
        score = 0.5  # Base score
        
        # ToT works well for problems with multiple solution paths
        if features['contains_constraints'] == 1:
            score += 0.3
            
        if features['potential_cognitive_trap'] == 1:
            score += 0.4  # ToT is excellent for problems that require considering multiple approaches
            
        # Higher score for complex problems with decision points
        if features['contains_uncertainty'] == 1:
            score += 0.3
            
        # Problems with comparison benefit from exploring multiple branches
        if features['contains_comparison'] == 1:
            score += 0.2
            
        # Pure math calculations might not benefit as much from ToT
        if features['primary_domain'] == 'math' and features['contains_constraints'] == 0:
            score -= 0.2
            
        return max(0.1, min(1.0, score))  # Ensure score is between 0.1 and 1.0
    
    def _calculate_got_score(self, features):
        """Calculate suitability score for Graph-of-Thought reasoning."""
        score = 0.5  # Base score
        
        # GoT works well for problems with interconnected concepts
        if features['primary_domain'] == 'logic':
            score += 0.4
            
        if features['primary_domain'] == 'complex':
            score += 0.3
            
        # Higher score for problems with multiple variables or relationships
        if features['contains_equations'] == 1 and features['algebra_score'] > 2:
            score += 0.2
            
        # Logic problems benefit from interconnected reasoning
        if features['contains_logic_keywords'] == 1:
            score += 0.3
            
        # Word problems with multiple entities and relationships
        if features['word_count'] > 50 and features['contains_math_operators'] == 1:
            score += 0.1
            
        return max(0.1, min(1.0, score))  # Ensure score is between 0.1 and 1.0


class DynamicTopologyRouter:
    """
    Routes problem statements to the optimal reasoning topology based on prediction.
    
    This component interfaces with the topology predictor and the reasoning topologies
    to dynamically select and apply the most suitable approach for each problem.
    """
    
    def __init__(self):
        """Initialize the router with the topology predictor and reasoning topologies."""
        self.predictor = TopologyPredictor()
        self.topologies = {}  # Will be populated with actual topology instances
        
    def register_topology(self, name, topology_instance):
        """
        Register a reasoning topology instance with the router.
        
        Parameters:
            name (str): Name identifier for the topology
            topology_instance: Instance of the reasoning topology class
        """
        self.topologies[name] = topology_instance
        
    def route_problem(self, problem_statement, force_topology=None):
        """
        Route a problem to the optimal reasoning topology.
        
        Parameters:
            problem_statement (str): The problem to solve
            force_topology (str, optional): Force the use of a specific topology
            
        Returns:
            dict: Contains the selected topology, prediction confidence, and routed response
        """
        # Predict the most suitable topology (unless forced)
        if force_topology and force_topology in self.topologies:
            selected_topology = force_topology
            prediction = {
                'predicted_topology': force_topology,
                'confidence': 1.0,
                'topology_scores': {force_topology: 1.0},
                'features': {}  # No features needed when forcing a topology
            }
        else:
            prediction = self.predictor.predict_topology(problem_statement)
            selected_topology = prediction['predicted_topology']
        
        # Ensure the selected topology is registered
        if selected_topology not in self.topologies:
            # Fallback to first available topology
            selected_topology = next(iter(self.topologies)) if self.topologies else None
            
        if not selected_topology:
            return {
                'error': 'No topology available for routing',
                'prediction': prediction
            }
        
        # Generate response using the selected topology
        topology_instance = self.topologies[selected_topology]
        response = topology_instance.generate_response(problem_statement)
        
        return {
            'selected_topology': selected_topology,
            'prediction_confidence': prediction['confidence'],
            'topology_scores': prediction['topology_scores'],
            'problem_features': prediction.get('features', {}),
            'response': response
        }


if __name__ == "__main__":
    # Test the feature extractor and predictor
    predictor = TopologyPredictor()
    
    test_problems = [
        "Solve the math problem: If 3x + 7 = 22, what is x?",
        "Solve this logic problem: All roses are flowers. Some flowers fade quickly. Therefore, some roses fade quickly. Is this conclusion valid?",
        "Solve this problem carefully: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "Solve this problem carefully: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?"
    ]
    
    print("Problem Topology Prediction Test:")
    for problem in test_problems:
        prediction = predictor.predict_topology(problem)
        print(f"\nProblem: {problem}")
        print(f"Predicted Topology: {prediction['predicted_topology']} (Confidence: {prediction['confidence']:.2f})")
        print(f"Topology Scores: {prediction['topology_scores']}")
        print(f"Key Features: {', '.join([f'{k}: {v}' for k, v in prediction['features'].items() if k in ['primary_domain', 'potential_cognitive_trap', 'contains_constraints']])}")