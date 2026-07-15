"""
Core dataset and data collator classes for EKG fine-tuning
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional


class EKGDataset(Dataset):
    """
    PyTorch Dataset for EKG training examples
    """
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 32768):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all examples
        self.tokenized_examples = []
        print(f"Tokenizing {len(examples)} examples...")
        
        for i, example in enumerate(examples):
            if i % 500 == 0:
                print(f"Tokenized {i}/{len(examples)} examples")
            
            tokenized = self._tokenize_example(example)
            if tokenized is not None:
                self.tokenized_examples.append(tokenized)
        
        print(f"Successfully tokenized {len(self.tokenized_examples)} examples")
    
    def _tokenize_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Tokenize a single example"""
        text = example['text']
        
        # Tokenize the full text
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # Skip if too short
        if len(encoded['input_ids']) < 10:
            return None
        
        # Create labels (same as input_ids for causal LM)
        labels = encoded['input_ids'].copy()
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'mace_label': torch.tensor(example['mace_label'], dtype=torch.long),
            'mortality_label': torch.tensor(example['mortality_label'], dtype=torch.long),
            'patient_id': example['patient_id'],
            'example_type': example['type']
        }
    
    def __len__(self):
        return len(self.tokenized_examples)
    
    def __getitem__(self, idx):
        return self.tokenized_examples[idx]


class EKGDataCollator:
    """
    Custom data collator for EKG training
    Handles padding and batching for our specific use case
    """
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        
        # Make sure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, features):
        # Extract the additional fields before padding
        mace_labels = []
        mortality_labels = []
        patient_ids = []
        example_types = []
        
        batch_features = []
        
        for feature in features:
            mace_labels.append(feature.pop('mace_label'))
            mortality_labels.append(feature.pop('mortality_label'))
            patient_ids.append(feature.pop('patient_id'))
            example_types.append(feature.pop('example_type'))
            batch_features.append(feature)
        
        # Pad the sequences
        batch = self.tokenizer.pad(
            batch_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the same as input_ids
        # but shifted during loss computation
        batch['labels'] = batch['input_ids'].clone()
        
        # Add back the additional fields
        batch['mace_labels'] = torch.stack(mace_labels)
        batch['mortality_labels'] = torch.stack(mortality_labels)
        batch['patient_ids'] = patient_ids
        batch['example_types'] = example_types
        
        return batch