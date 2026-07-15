"""
Main fine-tuning script for DeepSeek-R1-0528-Qwen3-8B on EKG data

Stage I: Frozen LLM abnormality detection (12 leads + 2 metadata groups)
Stage II: LoRA fine-tune for MACE/mortality risk prediction + causal explanations
"""

import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import wandb
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import Config, ModelConfig, DataConfig, TrainingConfig
from data.data_processor import EKGLeadBasedProcessor
from model.model import DeepSeekModelSetup
from model.stage1 import FrozenAbnormalityDetector, Stage1Cache
from data.data_module import create_data_module, EKGDataCollator
from model.trainer import EKGMedicalTrainer
from utils.common import set_seed, print_training_info, print_section_header, print_completion_message
from utils.evaluation import evaluate_model_on_test_set, compute_final_metrics_with_auc
from utils.reporting import (save_comprehensive_evaluation_summary, print_evaluation_results,
                           save_final_results)


def setup_wandb(config, run_name: str):
    """Initialize wandb logging"""
    try:
        wandb.init(
            project="deepseek-ekg-lead-finetune",
            name=run_name,
            config=config.__dict__,
            dir=config.training.output_dir
        )
        return True
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        return False


def run_stage1_and_build_examples(config, model_setup, tokenizer, cache_path=None):
    """
    Stage I: frozen LLM detects abnormalities from real patient features,
    then build Stage II training examples from those findings.
    """
    print_section_header("Stage I: Frozen LLM Abnormality Detection", 1)

    if model_setup.model is None:
        raise ValueError("Base model must be loaded before Stage I")

    detector = FrozenAbnormalityDetector(
        model=model_setup.model,
        tokenizer=tokenizer,
        max_new_tokens=config.model.stage1_max_new_tokens,
        temperature=config.model.stage1_temperature,
        top_p=config.model.top_p,
        do_sample=config.model.stage1_do_sample,
        max_input_length=config.model.stage1_max_input_length,
    )
    cache = Stage1Cache(cache_path) if cache_path else Stage1Cache(
        os.path.join(config.training.output_dir, "stage1_cache.json")
    )

    processor = EKGLeadBasedProcessor(config)
    df = processor.load_and_prepare_data(config.data.csv_path, config.data.max_samples)
    training_examples = processor.create_training_examples(df, detector=detector, cache=cache)

    detector.restore_adapters()

    print(f"Created {len(training_examples)} training examples after Stage I")
    if training_examples:
        example = training_examples[0]
        print(f"\nExample training data:")
        print(f"Type: {example['type']}")
        print(f"Text (first 300 chars): {example['text'][:300]}...")
        print(f"MACE: {example['mace_label']}, Mortality: {example['mortality_label']}")

    return training_examples


def setup_stage2_model(model_setup, config):
    """Stage II: prepare base model for LoRA fine-tuning"""
    print_section_header("Stage II: Setting up LoRA Fine-tuning", 2)

    model_setup.prepare_for_stage2_training()
    peft_model = model_setup.apply_lora()
    model_info = model_setup.get_model_info()
    print(f"Model info: {model_info}")
    return peft_model, model_info


def create_datasets(training_examples, tokenizer, config):
    """Create train and eval datasets"""
    print_section_header("Creating Datasets", 3)

    data_module = create_data_module(training_examples, tokenizer, config)
    train_dataset, eval_dataset = data_module.get_datasets()

    return train_dataset, eval_dataset


def setup_training(model_setup, peft_model, train_dataset, eval_dataset, tokenizer, config):
    """Setup training arguments and trainer"""
    print_section_header("Setting up Training", 4)

    training_args = model_setup.setup_training_args(config.training.output_dir, len(train_dataset))
    ekg_data_collator = EKGDataCollator(tokenizer, pad_to_multiple_of=8)

    trainer = EKGMedicalTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ekg_data_collator,
    )

    return trainer


def run_training(trainer, config, resume_from_checkpoint=None):
    """Execute the training process"""
    print_section_header("Training", 5)

    try:
        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


def save_model(model_setup, config):
    """Save the trained model"""
    print_section_header("Saving Model", 6)
    model_setup.save_model(config.training.output_dir)


def run_evaluation(peft_model, tokenizer, eval_dataset, config, title="Final Evaluation"):
    """Run comprehensive evaluation and generate reports"""
    print_section_header(title, 7 if "Final" in title else None)

    device = next(peft_model.parameters()).device
    predictions, probability_predictions, true_labels, generated_texts = evaluate_model_on_test_set(
        peft_model, tokenizer, eval_dataset, device, max_samples=100
    )

    final_metrics = compute_final_metrics_with_auc(
        predictions, probability_predictions, true_labels, config.training.output_dir
    )

    y_true_list = [
        [label['mace_label'] for label in true_labels],
        [label['mortality_label'] for label in true_labels]
    ]
    y_prob_list = [
        [p[0] for p in probability_predictions],
        [p[1] for p in probability_predictions]
    ]
    task_names = ['MACE', 'Mortality']

    save_comprehensive_evaluation_summary(
        final_metrics, y_true_list, y_prob_list, predictions,
        probability_predictions, task_names, config.training.output_dir
    )

    return final_metrics, predictions, probability_predictions, true_labels, generated_texts


def run_full_training(args, config, use_wandb):
    """Run full CLAIRE-α pipeline: Stage I (frozen) → Stage II (LoRA)"""
    # Load frozen base model for Stage I
    print_section_header("Loading Frozen Base Model", 0)
    model_setup = DeepSeekModelSetup(config)
    tokenizer = model_setup.setup_tokenizer()
    model_setup.load_base_model(prepare_for_training=False)

    # Stage I abnormality detection + example construction
    training_examples = run_stage1_and_build_examples(
        config, model_setup, tokenizer, cache_path=args.stage1_cache
    )

    # Stage II LoRA setup
    peft_model, model_info = setup_stage2_model(model_setup, config)

    # Create datasets
    train_dataset, eval_dataset = create_datasets(training_examples, tokenizer, config)

    # Setup training
    trainer = setup_training(model_setup, peft_model, train_dataset, eval_dataset, tokenizer, config)

    resume_from_checkpoint = None
    if args.resume:
        checkpoints = [d for d in os.listdir(config.training.output_dir)
                     if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from_checkpoint = os.path.join(config.training.output_dir, latest_checkpoint)
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    run_training(trainer, config, resume_from_checkpoint)
    save_model(model_setup, config)

    final_metrics, predictions, probability_predictions, true_labels, generated_texts = run_evaluation(
        peft_model, tokenizer, eval_dataset, config
    )

    print_evaluation_results(final_metrics, "FINAL TRAINING RESULTS")

    if use_wandb:
        wandb.log({"final_" + k: v for k, v in final_metrics.items() if v is not None})

    save_final_results(
        final_metrics, model_info, config.training.__dict__,
        predictions, probability_predictions, true_labels,
        generated_texts, config.training.output_dir
    )


def main():
    parser = argparse.ArgumentParser(
        description="CLAIRE-α: Stage I frozen abnormality detection + Stage II LoRA fine-tuning"
    )
    parser.add_argument("--csv_path", type=str, default="data/dataset/ekg.csv", help="Path to EKG CSV file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=30000, help="Maximum number of samples to use")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--stage1_cache",
        type=str,
        default=None,
        help="Optional path to Stage I JSON cache (load existing / save new findings)",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline_name = "claire_alpha"
    timestamped_output_dir = os.path.join(args.output_dir, f"{pipeline_name}_{timestamp}")

    config = Config(
        model=ModelConfig(),
        data=DataConfig(csv_path=args.csv_path, max_samples=args.max_samples),
        training=TrainingConfig(
            output_dir=timestamped_output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    )

    set_seed(config.seed)
    print_training_info(config)

    os.makedirs(config.training.output_dir, exist_ok=True)
    config.save_to_dir(config.training.output_dir)

    run_name = f"claire_alpha_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    use_wandb = setup_wandb(config, run_name)

    try:
        run_full_training(args, config, use_wandb)

        if use_wandb:
            wandb.finish()

        print_completion_message(config.training.output_dir)

    except Exception:
        if use_wandb:
            wandb.finish()
        raise


if __name__ == "__main__":
    main()
