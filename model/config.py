"""
Configuration management for DeepSeek EKG fine-tuning
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

# Hugging Face Hub id (used only when local weights are unavailable)
HF_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
LOCAL_MODEL_DIRNAME = "DeepSeek-R1-0528-Qwen3-8B"


def default_local_model_path() -> str:
    """Default on-disk weights: model/weights/DeepSeek-R1-0528-Qwen3-8B"""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "weights",
        LOCAL_MODEL_DIRNAME,
    )


def is_usable_local_model_dir(path: str) -> bool:
    """True if path looks like a complete Transformers model directory."""
    if not path or not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    has_tokenizer = (
        os.path.isfile(os.path.join(path, "tokenizer.json"))
        or os.path.isfile(os.path.join(path, "tokenizer_config.json"))
    )
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    has_weights = any(
        name.endswith(".safetensors") or name.endswith(".bin")
        for name in entries
    )
    return has_config and has_tokenizer and has_weights


def resolve_model_name(
    local_model_path: Optional[str] = None,
    hf_model_id: str = HF_MODEL_ID,
    explicit_model_name: Optional[str] = None,
) -> str:
    """
    Prefer local weights; fall back to Hugging Face Hub when local is missing/incomplete.

    Resolution order:
      1. explicit_model_name if it is a usable local directory
      2. local_model_path (default: model/weights/DeepSeek-...) if usable
      3. explicit_model_name if it is a Hub id / other non-dir string
      4. hf_model_id
    """
    local_path = os.path.abspath(local_model_path or default_local_model_path())

    if explicit_model_name and is_usable_local_model_dir(explicit_model_name):
        resolved = os.path.abspath(explicit_model_name)
        print(f"Using local model weights: {resolved}")
        return resolved

    if is_usable_local_model_dir(local_path):
        print(f"Using local model weights: {local_path}")
        return local_path

    if explicit_model_name and not os.path.isdir(explicit_model_name):
        print(
            f"Local weights not available at {local_path}; "
            f"using Hugging Face: {explicit_model_name}"
        )
        return explicit_model_name

    print(
        f"Local weights not available at {local_path}; "
        f"using Hugging Face: {hf_model_id}"
    )
    return hf_model_id


@dataclass
class ModelConfig:
    """Model configuration"""
    # Resolved source used by from_pretrained (local path or Hub id)
    model_name: Optional[str] = None
    # Preferred on-disk directory (checked first)
    local_model_path: Optional[str] = None
    # Hub fallback when local weights are not available
    hf_model_id: str = HF_MODEL_ID

    max_length: int = 32768
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Stage I (frozen abnormality detection) generation settings
    stage1_max_new_tokens: int = 256
    stage1_temperature: float = 0.3
    stage1_do_sample: bool = False
    stage1_max_input_length: int = 8192

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    def __post_init__(self):
        if self.local_model_path is None:
            self.local_model_path = default_local_model_path()
        else:
            self.local_model_path = os.path.abspath(self.local_model_path)

        # Always prefer local when present; Hub only as fallback
        self.model_name = resolve_model_name(
            local_model_path=self.local_model_path,
            hf_model_id=self.hf_model_id,
            explicit_model_name=self.model_name,
        )

        if self.lora_target_modules is None:
            # Common target modules for Qwen models
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

    @property
    def uses_local_weights(self) -> bool:
        return is_usable_local_model_dir(self.model_name or "")


@dataclass
class DataConfig:
    """Data configuration"""
    csv_path: str = ""
    max_samples: int = 1000
    train_split: float = 0.8
    eval_split: float = 0.1
    test_split: float = 0.1
    target_columns: Optional[List[str]] = None

    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = ['3p_MACE_binary', 'Mortality_Binary']


@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "./deepseek_ekg_finetune"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 5000
    save_steps: int = 5000
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Multi-GPU settings
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 4

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Optimization
    max_grad_norm: float = 1.0


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    seed: int = 42

    def save_to_dir(self, directory: str):
        """Save configuration to directory"""
        os.makedirs(directory, exist_ok=True)
        config_path = os.path.join(directory, "config.json")

        config_dict = asdict(self)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {config_path}")

    @classmethod
    def load_from_dir(cls, directory: str):
        """Load configuration from directory"""
        config_path = os.path.join(directory, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct nested dataclasses (ModelConfig re-resolves local vs Hub)
        model_config = ModelConfig(**config_dict['model'])
        data_config = DataConfig(**config_dict['data'])
        training_config = TrainingConfig(**config_dict['training'])

        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            seed=config_dict.get('seed', 42)
        )


# Default configuration instance
config = Config(
    model=ModelConfig(),
    data=DataConfig(),
    training=TrainingConfig()
)
