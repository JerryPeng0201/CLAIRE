"""
Standalone evaluation script for trained EKG models
Loads a trained model and generates comprehensive evaluation results including AUC curves
"""

import os
import sys
import argparse
import json
import torch
from datetime import datetime

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from model.config import Config, ModelConfig, DataConfig, TrainingConfig
from data.data_processor import EKGLeadBasedProcessor
from model.model import setup_model_and_tokenizer
from model.stage1 import FrozenAbnormalityDetector, Stage1Cache
from data.data_module import create_data_module
from utils.evaluation import evaluate_model_on_test_set, compute_final_metrics_with_auc
from utils.reporting import (save_comprehensive_evaluation_summary, print_evaluation_results,
                           save_evaluation_only_results)
from utils.common import set_seed, print_section_header


def load_trained_model(model_path, config):
    """
    Load a trained model from checkpoint directory
    
    Args:
        model_path: Path to the trained model directory
        config: Configuration object
        
    Returns:
        Tuple of (model, tokenizer, model_setup)
    """
    print_section_header("Loading Trained Model")
    
    # Setup base model and tokenizer (Stage II LoRA scaffold; adapters loaded below)
    model_setup, peft_model, tokenizer, _ = setup_model_and_tokenizer(config, for_training=True)
    
    # Check if model is properly initialized
    if model_setup.model is None:
        raise ValueError("Base model is not initialized")
    
    # Check if we have PEFT adapters
    adapter_path = os.path.join(model_path, "adapter_model.safetensors")
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_path) and os.path.exists(adapter_config_path):
        print("Loading PEFT adapters...")
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(model_setup.model, model_path)
        print(f"✅ PEFT adapters loaded from {model_path}")
    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        print("Loading full model checkpoint...")
        model_setup.model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
        peft_model = model_setup.model
        print(f"✅ Full model loaded from {model_path}")
    else:
        print(f"⚠️  No trained model found at {model_path}")
        print("Available files:")
        if os.path.exists(model_path):
            for file in os.listdir(model_path):
                print(f"  - {file}")
        print("Using base model without fine-tuning...")
        peft_model = model_setup.model
    
    return peft_model, tokenizer, model_setup


def load_evaluation_data(csv_path, max_samples, config, model, tokenizer, stage1_cache=None):
    """
    Load evaluation data with Stage I frozen-LLM abnormality detection
    (adapters disabled so Stage I matches the README frozen-base method).
    """
    print_section_header("Loading Evaluation Data (Stage I frozen LLM)")
    
    detector = FrozenAbnormalityDetector(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.model.stage1_max_new_tokens,
        temperature=config.model.stage1_temperature,
        top_p=config.model.top_p,
        do_sample=config.model.stage1_do_sample,
        max_input_length=config.model.stage1_max_input_length,
    )
    cache = Stage1Cache(stage1_cache) if stage1_cache else None

    processor = EKGLeadBasedProcessor(config)
    df = processor.load_and_prepare_data(csv_path, max_samples)
    training_examples = processor.create_training_examples(df, detector=detector, cache=cache)

    detector.restore_adapters()
    
    print(f"Loaded {len(training_examples)} examples from {csv_path}")
    
    risk_prediction_examples = [ex for ex in training_examples if ex['type'] == 'risk_prediction']
    mace_positive = sum(1 for ex in risk_prediction_examples if ex['mace_label'] == 1)
    mortality_positive = sum(1 for ex in risk_prediction_examples if ex['mortality_label'] == 1)
    
    print(f"Risk prediction examples: {len(risk_prediction_examples)}")
    if risk_prediction_examples:
        print(f"MACE positive cases: {mace_positive} ({mace_positive/len(risk_prediction_examples)*100:.1f}%)")
        print(f"Mortality positive cases: {mortality_positive} ({mortality_positive/len(risk_prediction_examples)*100:.1f}%)")
    
    return training_examples


def create_evaluation_dataset(training_examples, tokenizer, config):
    """
    Create evaluation dataset from training examples
    
    Args:
        training_examples: List of training examples
        tokenizer: Tokenizer
        config: Configuration object
        
    Returns:
        Evaluation dataset
    """
    print_section_header("Creating Evaluation Dataset")
    
    # For evaluation, we'll use all examples as "eval" dataset
    data_module = create_data_module(training_examples, tokenizer, config)
    _, eval_dataset = data_module.get_datasets()
    
    print(f"Created evaluation dataset with {len(eval_dataset)} examples")
    
    return eval_dataset


def run_comprehensive_evaluation(model, tokenizer, eval_dataset, output_dir, max_samples=200):
    """
    Run comprehensive evaluation and generate all outputs
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        output_dir: Output directory for results
        max_samples: Maximum samples to evaluate
        
    Returns:
        Dictionary of final metrics
    """
    print_section_header("Running Comprehensive Evaluation")
    
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # Run evaluation
    predictions, probability_predictions, true_labels, generated_texts = evaluate_model_on_test_set(
        model, tokenizer, eval_dataset, device, max_samples=max_samples
    )
    
    print(f"Evaluated {len(predictions)} samples")
    
    # Compute comprehensive metrics with AUC curves
    final_metrics = compute_final_metrics_with_auc(
        predictions, probability_predictions, true_labels, output_dir
    )
    
    # Generate comprehensive reporting
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
        probability_predictions, task_names, output_dir
    )
    
    # Save detailed results
    save_evaluation_only_results(
        final_metrics, predictions, probability_predictions, 
        true_labels, generated_texts, output_dir
    )
    
    return final_metrics


def save_evaluation_metadata(args, config, output_dir):
    """Save metadata about the evaluation run"""
    metadata = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'data_path': args.csv_path,
            'max_samples': args.max_samples,
            'output_dir': output_dir,
            'config_used': {
                'model_name': config.model.model_name,
                'max_length': config.model.max_length,
                'seed': config.seed
            }
        },
        'arguments': vars(args)
    }
    
    metadata_path = os.path.join(output_dir, 'evaluation_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Evaluation metadata saved to {metadata_path}")


def print_evaluation_summary(final_metrics, output_dir):
    """Print a comprehensive evaluation summary"""
    print("\n" + "="*80)
    print("🎯 EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Print key metrics
    print_evaluation_results(final_metrics, "EVALUATION RESULTS")
    
    # Print file locations
    print(f"\n📁 All results saved to: {output_dir}")
    print("\n📊 Generated Files:")
    print("   ROC Curves:")
    print("   • mace_roc_curve.png/pdf")
    print("   • mortality_roc_curve.png/pdf") 
    print("   • combined_roc_curves.png/pdf")
    print("\n   Data Files:")
    print("   • mace_roc_data.json, mace_predictions.csv")
    print("   • mortality_roc_data.json, mortality_predictions.csv")
    print("\n   Summary Files:")
    print("   • evaluation_summary.json/csv")
    print("   • evaluation_results.json")
    print("   • evaluation_metadata.json")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained EKG model and generate comprehensive results")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model directory (containing adapter_model.safetensors or pytorch_model.bin)")
    parser.add_argument("--csv_path", type=str, default="CLAIRE_Alpha/data/dataset/ekg.csv", 
                       help="Path to EKG CSV file for evaluation")
    parser.add_argument("--output_dir", type=str, default="CLAIRE_Alpha/eval_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--max_samples", type=int, default=200,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--create_timestamped_dir", action="store_true",
                       help="Create timestamped subdirectory for results")
    parser.add_argument(
        "--stage1_cache",
        type=str,
        default=None,
        help="Optional Stage I JSON cache path (reuse frozen-LLM findings)",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.csv_path):
        print(f"❌ Error: CSV path does not exist: {args.csv_path}")
        sys.exit(1)
    
    # Create output directory
    if args.create_timestamped_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("🔬 EKG MODEL EVALUATION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.csv_path}")
    print(f"Max samples: {args.max_samples}")
    print(f"Output: {output_dir}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
    print("="*80)
    
    # Setup configuration (using defaults for evaluation)
    config = Config(
        model=ModelConfig(),
        data=DataConfig(csv_path=args.csv_path, max_samples=args.max_samples),
        training=TrainingConfig(output_dir=output_dir)
    )
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    try:
        # Step 1: Load trained model
        model, tokenizer, _model_setup = load_trained_model(args.model_path, config)
        
        # Step 2: Load evaluation data via Stage I frozen LLM (adapters off)
        training_examples = load_evaluation_data(
            args.csv_path,
            args.max_samples,
            config,
            model=model,
            tokenizer=tokenizer,
            stage1_cache=args.stage1_cache,
        )
        
        # Step 3: Create evaluation dataset
        eval_dataset = create_evaluation_dataset(training_examples, tokenizer, config)
        
        # Step 4: Run comprehensive evaluation
        final_metrics = run_comprehensive_evaluation(
            model, tokenizer, eval_dataset, output_dir, max_samples=args.max_samples
        )
        
        # Step 5: Save metadata
        save_evaluation_metadata(args, config, output_dir)
        
        # Step 6: Print summary
        print_evaluation_summary(final_metrics, output_dir)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()