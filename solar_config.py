"""
SOLAR Framework Configuration Module

This module handles loading and accessing configuration settings from config.ini
"""

import os
import configparser

class SolarConfig:
    """Configuration handler for SOLAR framework"""
    
    def __init__(self):
        """Initialize configuration by reading from config.ini"""
        self.config = configparser.ConfigParser()
        
        # Default config path
        self.config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        
        # Load configuration
        self.load_config()
    
    def load_config(self):
        """Load configuration from config file"""
        if os.path.exists(self.config_path):
            self.config.read(self.config_path)
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration if config file doesn't exist"""
        self.config['MODEL'] = {
            'ollama_model': 'qwq'
        }
        self.config['BENCHMARK'] = {
            'problems_per_category': '2',
            'num_runs': '2'
        }
        
        # Write default config to file
        with open(self.config_path, 'w') as f:
            self.config.write(f)
    
    def get_ollama_model(self):
        """Get the configured Ollama model name"""
        return self.config.get('MODEL', 'ollama_model', fallback='qwq')
    
    def set_ollama_model(self, model_name):
        """Set the Ollama model name and save to config"""
        if not self.config.has_section('MODEL'):
            self.config.add_section('MODEL')
        
        self.config.set('MODEL', 'ollama_model', model_name)
        
        # Save changes to config file
        with open(self.config_path, 'w') as f:
            self.config.write(f)
    
    def get_problems_per_category(self):
        """Get the configured number of problems per category"""
        return self.config.getint('BENCHMARK', 'problems_per_category', fallback=2)
    
    def get_num_runs(self):
        """Get the configured number of runs per problem"""
        return self.config.getint('BENCHMARK', 'num_runs', fallback=2)

# Create a singleton instance for easy import
config = SolarConfig()

# For direct import
def get_ollama_model():
    """Get the configured Ollama model name"""
    return config.get_ollama_model()

def set_ollama_model(model_name):
    """Set the Ollama model name"""
    config.set_ollama_model(model_name)