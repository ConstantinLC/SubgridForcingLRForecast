import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for training the forecasting model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def load_configuration(config_file):
    with open(config_file, 'r') as config_stream:
        config = yaml.safe_load(config_stream)
    return config