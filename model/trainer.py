"""
Custom trainer for EKG medical evaluation
"""

import torch
import numpy as np
from transformers.trainer import Trainer
from utils.metrics import calculate_medical_metrics, extract_predictions_from_text


class EKGMedicalTrainer(Trainer):
    """
    Custom trainer with medical evaluation metrics
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss
        """
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**{k: v for k, v in inputs.items() 
                          if k not in ['mace_labels', 'mortality_labels', 'patient_ids', 'example_types']})
        
        # Compute language modeling loss
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and compute medical metrics
        """
        # Standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add medical metrics if evaluation dataset is not too large
        if eval_dataset is not None and len(eval_dataset) <= 100:
            try:
                medical_metrics = self._compute_medical_metrics(eval_dataset, max_samples=50)
                eval_results.update({f"{metric_key_prefix}_{k}": v for k, v in medical_metrics.items()})
            except Exception as e:
                print(f"Could not compute medical metrics: {e}")
        
        return eval_results
    
    def _compute_medical_metrics(self, eval_dataset, max_samples=50):
        """
        Generate predictions and compute medical metrics
        """
        model = self.model
        tokenizer = self.tokenizer
        device = next(model.parameters()).device
        
        if tokenizer is None:
            raise ValueError("Tokenizer is not initialized")
            
        if model is None:
            raise ValueError("Model is not initialized")
        
        model.eval()
        predictions = []
        true_mace_labels = []
        true_mortality_labels = []
        
        print(f"Generating predictions for medical metrics evaluation...")
        
        # Sample a subset for evaluation
        sample_size = min(max_samples, len(eval_dataset))
        indices = np.random.choice(len(eval_dataset), size=sample_size, replace=False)
        
        for i, idx in enumerate(indices):
            sample = eval_dataset[idx]
            
            # Only evaluate risk prediction examples
            if sample.get('example_type') != 'risk_prediction':
                continue
            
            # Get the input text
            input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            
            # Find the assistant prompt
            assistant_pos = input_text.find('<|Assistant|>')
            if assistant_pos == -1:
                continue
            
            prompt = input_text[:assistant_pos + len('<|Assistant|>')]
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
            if torch.cuda.is_available():
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Extract predictions
            mace_pred, mortality_pred = extract_predictions_from_text(generated_text)
            
            predictions.append((mace_pred, mortality_pred))
            true_mace_labels.append(sample['mace_label'].item())
            true_mortality_labels.append(sample['mortality_label'].item())
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1} predictions...")
        
        if len(predictions) == 0:
            return {}
        
        # Calculate medical metrics
        mace_preds = [p[0] for p in predictions]
        mortality_preds = [p[1] for p in predictions]
        
        # MACE metrics
        mace_metrics = calculate_medical_metrics(true_mace_labels, mace_preds, "mace")
        
        # Mortality metrics
        mortality_metrics = calculate_medical_metrics(true_mortality_labels, mortality_preds, "mortality")
        
        # Combined metrics
        mace_auc = mace_metrics.get('mace_auc')
        mort_auc = mortality_metrics.get('mortality_auc')
        mean_auc = None
        if mace_auc is not None and mort_auc is not None:
            mean_auc = (mace_auc + mort_auc) / 2

        combined_metrics = {
            'mean_auc': mean_auc,
            'mean_sensitivity': (mace_metrics.get('mace_sensitivity', 0) + mortality_metrics.get('mortality_sensitivity', 0)) / 2,
            'mean_specificity': (mace_metrics.get('mace_specificity', 0) + mortality_metrics.get('mortality_specificity', 0)) / 2,
        }
        
        # Combine all metrics
        all_metrics = {**mace_metrics, **mortality_metrics, **combined_metrics}
        
        mace_auc_str = f"{mace_auc:.3f}" if mace_auc is not None else "N/A"
        mort_auc_str = f"{mort_auc:.3f}" if mort_auc is not None else "N/A"
        mean_auc_str = f"{mean_auc:.3f}" if mean_auc is not None else "N/A"
        print(f"Medical Metrics - MACE AUC: {mace_auc_str}, "
              f"Mortality AUC: {mort_auc_str}, "
              f"Mean AUC: {mean_auc_str}")
        
        return all_metrics
    
    def log(self, logs, start_time=None):
        """
        Enhanced logging with GPU memory tracking
        """
        super().log(logs)
        
        # Add GPU memory logging
        if torch.cuda.is_available() and hasattr(self, 'state') and self.state.is_local_process_zero:
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                logs.update({
                    f'gpu_{i}_memory_allocated_gb': memory_allocated,
                    f'gpu_{i}_memory_reserved_gb': memory_reserved
                })