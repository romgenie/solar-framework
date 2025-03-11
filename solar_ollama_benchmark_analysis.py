"""
SOLAR Framework Benchmark Analysis

This script analyzes the results of the SOLAR benchmark against Ollama models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def load_benchmark_results(file_path='solar_benchmark_results.json'):
    """Load benchmark results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_latency(results):
    """Analyze latency metrics from benchmark results."""
    solar_latency = [result['avg_latency'] for result in results['solar']]
    hybrid_latency = [result['avg_latency'] for result in results['hybrid']]
    
    ollama_latency = {}
    for model in results['ollama']:
        ollama_latency[model] = [result['avg_latency'] for result in results['ollama'][model]]
    
    # Calculate average latency
    avg_solar = np.mean(solar_latency)
    avg_hybrid = np.mean(hybrid_latency)
    avg_ollama = {model: np.mean(latencies) for model, latencies in ollama_latency.items()}
    
    # Convert microseconds to milliseconds for SOLAR components
    avg_solar_ms = avg_solar * 1000
    avg_hybrid_ms = avg_hybrid * 1000
    
    print("\n=== Latency Analysis ===")
    print(f"SOLAR Topological Rewarding: {avg_solar_ms:.4f} ms")
    print(f"SOLAR Hybrid Scaling: {avg_hybrid_ms:.4f} ms")
    for model, avg in avg_ollama.items():
        print(f"Ollama {model}: {avg*1000:.2f} ms")
    
    # Create latency comparison table
    headers = ["Problem"]
    solar_row = ["SOLAR (ms)"]
    hybrid_row = ["Hybrid (ms)"]
    ollama_rows = {model: [f"{model} (ms)"] for model in ollama_latency}
    
    for i, problem in enumerate(results['solar']):
        problem_text = problem['problem']
        headers.append(f"Problem {i+1}")
        solar_row.append(f"{solar_latency[i]*1000:.2f}")
        hybrid_row.append(f"{hybrid_latency[i]*1000:.2f}")
        
        for model in ollama_latency:
            ollama_rows[model].append(f"{ollama_latency[model][i]*1000:.2f}")
    
    table_rows = [solar_row, hybrid_row] + list(ollama_rows.values())
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="grid"))
    
    return {
        'solar': avg_solar,
        'hybrid': avg_hybrid,
        'ollama': avg_ollama
    }

def analyze_topology_distribution(results):
    """Analyze topology selection distribution."""
    topology_counts = {}
    for result in results['solar']:
        topo = result['selected_topology']
        topology_counts[topo] = topology_counts.get(topo, 0) + 1
    
    total = len(results['solar'])
    
    print("\n=== Topology Selection Analysis ===")
    table_data = []
    for topo, count in topology_counts.items():
        percentage = (count/total) * 100
        table_data.append([topo, count, f"{percentage:.1f}%"])
    
    print(tabulate(table_data, headers=["Topology", "Count", "Percentage"], tablefmt="grid"))
    
    return topology_counts

def plot_latency_comparison(latency_data):
    """Generate latency comparison plot."""
    models = ['SOLAR', 'Hybrid'] + list(latency_data['ollama'].keys())
    latencies = [latency_data['solar']*1000, latency_data['hybrid']*1000] + [latency_data['ollama'][model]*1000 for model in latency_data['ollama']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, latencies, color=['blue', 'green'] + ['red'] * len(latency_data['ollama']))
    
    # Add latency values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} ms',
                ha='center', va='bottom', rotation=0)
    
    plt.title('Average Latency Comparison')
    plt.ylabel('Latency (milliseconds)')
    plt.xlabel('Model')
    plt.yscale('log')  # Use log scale due to large differences
    plt.grid(True, alpha=0.3)
    plt.savefig('latency_comparison.png')
    print("\nLatency comparison plot saved as 'latency_comparison.png'")

def analyze_response_quality(results):
    """Simple response quality analysis."""
    print("\n=== Response Analysis ===")
    
    # Compare a random sample
    sample_idx = np.random.randint(0, len(results['solar']))
    problem = results['solar'][sample_idx]['problem']
    solar_response = results['solar'][sample_idx]['response']
    hybrid_response = results['hybrid'][sample_idx]['response']
    
    ollama_responses = {}
    for model in results['ollama']:
        ollama_responses[model] = results['ollama'][model][sample_idx]['response']
    
    print(f"Sample Problem: {problem}")
    print(f"SOLAR Response: {solar_response}")
    print(f"Hybrid Response: {hybrid_response}")
    for model, response in ollama_responses.items():
        print(f"{model} Response: {response[:150]}...")
    
    # Word count comparison
    print("\n=== Response Word Count Comparison ===")
    solar_word_counts = [len(result['response'].split()) for result in results['solar']]
    hybrid_word_counts = [len(result['response'].split()) for result in results['hybrid']]
    
    ollama_word_counts = {}
    for model in results['ollama']:
        ollama_word_counts[model] = [len(result['response'].split()) for result in results['ollama'][model]]
    
    avg_solar_words = np.mean(solar_word_counts)
    avg_hybrid_words = np.mean(hybrid_word_counts)
    avg_ollama_words = {model: np.mean(counts) for model, counts in ollama_word_counts.items()}
    
    word_count_data = [
        ["SOLAR", avg_solar_words],
        ["Hybrid", avg_hybrid_words]
    ]
    for model, avg in avg_ollama_words.items():
        word_count_data.append([model, avg])
    
    print(tabulate(word_count_data, headers=["Model", "Avg Word Count"], tablefmt="grid"))

if __name__ == "__main__":
    results = load_benchmark_results()
    print("Loaded benchmark results from solar_benchmark_results.json")
    
    # Analyze latency
    latency_data = analyze_latency(results)
    
    # Analyze topology distribution
    topology_counts = analyze_topology_distribution(results)
    
    # Plot latency comparison
    plot_latency_comparison(latency_data)
    
    # Analyze response quality
    analyze_response_quality(results)
    
    print("\nAnalysis complete!")