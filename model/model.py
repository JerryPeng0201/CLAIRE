"""
Model setup for DeepSeek-R1-0528-Qwen3-8B with LoRA fine-tuning
"""

import os
import torch

from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import warnings
warnings.filterwarnings('ignore')


def _local_files_only(model_source: str) -> bool:
    """Avoid Hub network calls when loading from a local weights directory."""
    return os.path.isdir(model_source)


class DeepSeekModelSetup:
    """
    Setup DeepSeek-R1-0528-Qwen3-8B model with LoRA configuration
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_tokenizer(self):
        """Load and configure tokenizer"""
        model_source = self.config.model.model_name
        print(f"Loading tokenizer: {model_source}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=_local_files_only(model_source),
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set up special tokens for structured reasoning
        special_tokens = {
            "additional_special_tokens": [
                "<think>", "</think>", 
                "<cause>", "</cause>",
                "<intermediate effect>", "</intermediate effect>",
                "<effect>", "</effect>"
            ]
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens")
        
        return self.tokenizer
    
    def setup_quantization_config(self):
        """Setup quantization configuration for memory efficiency"""
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            return quantization_config
        return None
    
    def load_base_model(self, prepare_for_training: bool = False):
        """
        Load the base DeepSeek model.

        Args:
            prepare_for_training: If True, prepare for k-bit LoRA training (Stage II).
                Stage I uses the frozen base with prepare_for_training=False.
        """
        model_source = self.config.model.model_name
        print(f"Loading base model: {model_source}")
        
        quantization_config = self.setup_quantization_config()
        
        # Model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "local_files_only": _local_files_only(model_source),
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **model_kwargs
        )
        
        # Resize token embeddings if we added special tokens
        if self.tokenizer is not None:  
            if hasattr(self.tokenizer, 'added_tokens_encoder') and len(self.tokenizer.added_tokens_encoder) > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        if prepare_for_training and quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        print(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
        return self.model

    def prepare_for_stage2_training(self):
        """Prepare the frozen Stage I base model for Stage II LoRA fine-tuning."""
        if self.model is None:
            raise ValueError("Base model must be loaded first")
        if torch.cuda.is_available():
            self.model = prepare_model_for_kbit_training(self.model)
        return self.model
    
    def setup_lora_config(self):
        """Setup LoRA configuration"""
        print("Setting up LoRA configuration...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=self.config.model.lora_target_modules,
            bias="none",
        )
        
        print(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
        print(f"Target modules: {lora_config.target_modules}")
        
        return lora_config
    
    def apply_lora(self):
        """Apply LoRA to the model"""
        if self.model is None:
            raise ValueError("Base model must be loaded first")
            
        lora_config = self.setup_lora_config()
        
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters info
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def setup_training_args(self, output_dir: str, train_dataset_size: int):
        """Setup training arguments"""
        # Calculate total steps for proper scheduling
        total_steps = (
            train_dataset_size 
            // self.config.training.per_device_train_batch_size 
            // self.config.training.gradient_accumulation_steps 
            * self.config.training.num_train_epochs
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            eval_strategy=self.config.training.evaluation_strategy,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            
            # Multi-GPU settings
            ddp_find_unused_parameters=self.config.training.ddp_find_unused_parameters,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            
            # Mixed precision
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            
            # Memory optimization
            gradient_checkpointing=True,
            max_grad_norm=self.config.training.max_grad_norm,
            
            # Logging and saving
            report_to=["tensorboard"],
            logging_dir=f"{output_dir}/logs",
            
            # Remove unused columns to save memory
            remove_unused_columns=False,
            
            # Data loading
            dataloader_pin_memory=True,
        )
        
        print(f"Training arguments configured for {total_steps} total steps")
        return training_args
    
    def setup_data_collator(self):
        """Setup data collator for language modeling"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be setup first")
            
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked
            pad_to_multiple_of=8,  # For efficiency on modern GPUs
        )
        
        return data_collator
    
    def get_model_info(self):
        """Get model information"""
        if self.peft_model is None:
            raise ValueError("Model must be setup first")
        
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be setup first")
        
        info = {
            "model_name": self.config.model.model_name,
            "hf_model_id": getattr(self.config.model, "hf_model_id", None),
            "local_model_path": getattr(self.config.model, "local_model_path", None),
            "uses_local_weights": getattr(self.config.model, "uses_local_weights", False),
            "total_params": self.model.num_parameters(),
            "trainable_params": sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad),
            "vocab_size": len(self.tokenizer),
            "max_length": self.config.model.max_length,
        }
        
        info["trainable_percentage"] = (info["trainable_params"] / info["total_params"]) * 100
        
        return info
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned model"""
        if self.peft_model is None:
            raise ValueError("Model must be setup and trained first")
            
        print(f"Saving model to {output_dir}")
        
        # Save the LoRA adapters
        self.peft_model.save_pretrained(output_dir)
        
        # Save the tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        print("Model saved successfully")
        
    def load_for_inference(self, adapter_path: str):
        """Load model for inference with proper tokenizer and embedding sizing"""
        from peft import PeftModel
        
        print(f"Loading model for inference from {adapter_path}")
        model_source = self.config.model.model_name
        
        # Load tokenizer first (this has the correct vocabulary size)
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, 
            trust_remote_code=True,
            padding_side="right"
        )
        
        print(f"Loaded tokenizer with {len(tokenizer)} tokens")
        
        # Load base model (local path preferred via config resolution)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=_local_files_only(model_source),
        )
        
        print(f"Base model original vocab size: {base_model.config.vocab_size}")
        
        # CRITICAL: Resize embeddings to match the saved tokenizer
        if len(tokenizer) != base_model.config.vocab_size:
            print(f"Resizing embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))
        
        # Now load LoRA adapters (this should work without size mismatch)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        print(f"Successfully loaded model with {len(tokenizer)} vocabulary size")
        
        return model, tokenizer
    
    def generate_response(self, model, tokenizer, prompt: str, max_new_tokens: int = 512):
        """Generate response for inference"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.model.max_length)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                do_sample=self.config.model.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        return response

def setup_model_and_tokenizer(config, for_training: bool = True):
    """
    Convenience function to setup model and tokenizer.

    Args:
        for_training: If True, prepare for k-bit training and apply LoRA (Stage II).
            If False, load frozen base only (Stage I / inference prep).
    """
    model_setup = DeepSeekModelSetup(config)
    
    # Setup tokenizer
    tokenizer = model_setup.setup_tokenizer()
    
    # Load base model (optionally prepare for Stage II training)
    model_setup.load_base_model(prepare_for_training=for_training)

    peft_model = None
    data_collator = None
    if for_training:
        peft_model = model_setup.apply_lora()
        data_collator = model_setup.setup_data_collator()
    
    return model_setup, peft_model, tokenizer, data_collator