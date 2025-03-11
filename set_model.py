#!/usr/bin/env python3
"""
SOLAR Framework Model Configuration Utility

This script sets the Ollama model to use for all benchmarks.
It updates the config.ini file that is used by all components.
"""

import argparse
import sys
from solar_config import config, set_ollama_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set the Ollama model for SOLAR framework benchmarks")
    
    parser.add_argument("model", type=str, help="Name of the Ollama model to use")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Get current model
    current_model = config.get_ollama_model()
    
    # Set new model
    set_ollama_model(args.model)
    
    print(f"SOLAR Framework Ollama model updated:")
    print(f"  Previous model: {current_model}")
    print(f"  New model: {args.model}")
    print("")
    print("This change will affect all benchmark runs. To use a different model for a single run,")
    print("use the --ollama-model parameter with run_standard_benchmarks.py")

if __name__ == "__main__":
    main()