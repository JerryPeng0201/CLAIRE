"""
High-level data module for managing EKG datasets
"""

from typing import List, Any, Dict
from .dataset import EKGDataset, EKGDataCollator
from .data_processor import EKGLeadBasedProcessor


class EKGDataModule:
    """
    Data module for managing EKG datasets
    """

    def __init__(
        self,
        train_examples: List[Dict[str, Any]],
        eval_examples: List[Dict[str, Any]],
        tokenizer,
        config,
    ):
        self.train_examples = train_examples
        self.eval_examples = eval_examples
        self.tokenizer = tokenizer
        self.config = config

        self.train_dataset = EKGDataset(
            train_examples, tokenizer, config.model.max_length
        )
        self.eval_dataset = EKGDataset(
            eval_examples, tokenizer, config.model.max_length
        )

        print(f"Created train dataset with {len(self.train_dataset)} examples")
        print(f"Created eval dataset with {len(self.eval_dataset)} examples")

    def get_datasets(self):
        """Get train and eval datasets"""
        return self.train_dataset, self.eval_dataset

    def get_data_collator(self):
        """Get data collator"""
        return EKGDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8
        )


def create_data_module(training_examples: List[Dict[str, Any]], tokenizer, config):
    """
    Factory function to create data module with train/eval split
    """
    processor = EKGLeadBasedProcessor(config)
    train_examples, eval_examples, _ = processor.split_data(training_examples)

    return EKGDataModule(train_examples, eval_examples, tokenizer, config)
