"""
Common utility functions
"""

import random
import torch
import numpy as np
from typing import Optional


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_training_info(config):
    """Print training configuration information"""
    print("=" * 60)
    print("DeepSeek-R1-0528-Qwen3-8B EKG Lead-Based Fine-tuning")
    print("=" * 60)
    print(f"Model: {config.model.model_name}")
    print(f"Dataset: {config.data.csv_path}")
    print(f"Max samples: {config.data.max_samples}")
    print(f"Output directory: {config.training.output_dir}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Batch size per device: {config.training.per_device_train_batch_size}")
    print(f"Effective batch size: {config.training.per_device_train_batch_size * torch.cuda.device_count() * config.training.gradient_accumulation_steps}")
    print(f"Generate AUC curves: True (always enabled)")


def print_section_header(title: str, step_number: Optional[int] = None):
    """Print a formatted section header"""
    header = f"Step {step_number}: {title}" if step_number else title
    print("\n" + "=" * 30)
    print(header)
    print("=" * 30)


def print_completion_message(output_dir: str):
    """Print completion message with output directory"""
    print("\n" + "=" * 60)
    print("Fine-tuning completed successfully!")
    print(f"Separate MACE and Mortality AUC curves and data files saved to: {output_dir}")
    print("=" * 60)