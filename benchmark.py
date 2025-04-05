# benchmark.py

import yaml
import logging

from data.data_loaders import get_dataloader
from models.sam_model import SAMModel
from evaluation.evaluator import Evaluator
from utils.logging_utils import setup_logging

def load_config(config_path="config/benchmark_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    setup_logging(config.get("logging", {}))
    logging.info("Configuration loaded successfully.")

    # Initialize the SAM model
    model = SAMModel(config["model"])

    # Process each dataset defined in the configuration
    for dataset in config["datasets"]:
        ds_name = dataset["name"]
        logging.info(f"Starting evaluation for dataset: {ds_name}")

        # Obtain data loaders for train, validation, and test splits
        train_loader, val_loader, test_loader = get_dataloader(
            dataset, config["augmentation"], config["hyperparameters"]
        )

        # Initialize evaluator for the current dataset
        evaluator = Evaluator(model, dataset, config)

        # Run the evaluation pipeline
        evaluator.run_evaluation(train_loader, val_loader, test_loader)
        logging.info(f"Completed evaluation for dataset: {ds_name}")

    logging.info("Benchmarking completed successfully.")

if __name__ == "__main__":
    main()
