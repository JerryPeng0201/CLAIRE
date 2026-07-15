"""
CLAIRE-α patient-level inference

For each patient:
  Stage I  — frozen base LLM analyzes EKG features (12 leads + 2 metadata groups)
  Stage II — LoRA fine-tuned model immediately predicts MACE/mortality with explanations

Defaults:
  --model_path  projects/UH/CLAIRE_Alpha_local/checkpoints/claire_alpha_20250814_141439
  --csv_path    projects/UH/CLAIRE_Alpha_local/data/dataset/ekg.csv
  --num_patients 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from peft import PeftModel

# Repo root on PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import Config, ModelConfig, DataConfig, TrainingConfig
from model.model import DeepSeekModelSetup
from model.stage1 import FrozenAbnormalityDetector, Stage1Cache
from data.data_processor import EKGLeadBasedProcessor
from utils.metrics import extract_predictions_from_text
from utils.evaluation import extract_probabilities_from_text
from utils.common import set_seed, print_section_header


# scripts/ → CLAIRE-Alpha/ → github/ → UH/
_UH_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_DEFAULT_MODEL = os.path.join(
    _UH_ROOT,
    "CLAIRE_Alpha_local/checkpoints/claire_alpha_20250814_141439",
)
_DEFAULT_CSV = os.path.join(
    _UH_ROOT,
    "CLAIRE_Alpha_local/data/dataset/ekg.csv",
)


def _resolve_path(path: str) -> str:
    """Expand user/relative paths; keep absolute paths as-is."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    # Also try under UH root (projects/UH/...)
    alt = os.path.join(_UH_ROOT, path)
    if os.path.exists(alt):
        return alt
    # Try full projects-relative path from workspace user root
    user_root = os.path.abspath(os.path.join(_UH_ROOT, "..", ".."))
    alt2 = os.path.join(user_root, path)
    return alt2 if os.path.exists(alt2) else cwd_path


def load_finetuned_model(model_path: str, config: Config):
    """Load local base weights + LoRA adapters for inference."""
    print_section_header("Loading fine-tuned model")
    model_setup = DeepSeekModelSetup(config)
    # Prefer checkpoint tokenizer if present (special tokens from training)
    tokenizer_src = model_path if os.path.exists(
        os.path.join(model_path, "tokenizer_config.json")
    ) else config.model.model_name

    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from model.model import _local_files_only

    print(f"Loading tokenizer from: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=_local_files_only(tokenizer_src),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_setup.tokenizer = tokenizer

    model_setup.load_base_model(prepare_for_training=False)

    adapter_weights = os.path.join(model_path, "adapter_model.safetensors")
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if not (os.path.exists(adapter_weights) and os.path.exists(adapter_cfg)):
        raise FileNotFoundError(
            f"LoRA adapters not found under {model_path} "
            f"(need adapter_model.safetensors + adapter_config.json)"
        )

    # Align embeddings if tokenizer grew during fine-tuning
    if len(tokenizer) != model_setup.model.config.vocab_size:
        print(
            f"Resizing embeddings: {model_setup.model.config.vocab_size} → {len(tokenizer)}"
        )
        model_setup.model.resize_token_embeddings(len(tokenizer))

    print(f"Loading LoRA adapters from {model_path}")
    model = PeftModel.from_pretrained(model_setup.model, model_path)
    model.eval()
    print("Fine-tuned model ready")
    return model, tokenizer, model_setup


def build_stage1_lead_prompt(processor: EKGLeadBasedProcessor, patient_data, lead: str, features: List[str]) -> str:
    base = processor.format_lead_data(patient_data, lead, features)
    return (
        base
        + "\nIn your answer, (1) state abnormal or normal findings, and "
        "(2) briefly explain *why* using the measurement values provided.\n"
    )


def build_stage1_demo_prompt(processor: EKGLeadBasedProcessor, patient_data) -> str:
    return (
        processor.format_demographics_group_prompt(patient_data)
        + "\nExplain briefly how these demographics may relate to cardiovascular risk.\n"
    )


def build_stage1_clinical_prompt(processor: EKGLeadBasedProcessor, patient_data) -> str:
    return (
        processor.format_clinical_metadata_group_prompt(patient_data)
        + "\nExplain briefly how this clinical/acquisition context may matter for risk.\n"
    )


def build_stage2_prompt(processor: EKGLeadBasedProcessor, patient_data, aggregated_findings: str) -> str:
    """Stage II prompt that asks for explicit why-explanations for MACE and mortality."""
    demographics = processor._format_demographics(patient_data)
    return f"""Patient Demographics:
{demographics}

{aggregated_findings}

Based on the patient demographics and EKG abnormalities identified above (Stage I), please assess the risk for:
1. Major Adverse Cardiac Events (MACE) within 3 years
2. Mortality risk

Provide:
- Binary predictions (0 = No, 1 = Yes)
- Estimated probabilities (0.0-1.0)
- Clear clinical reasoning explaining *why* you predict MACE yes/no
- Clear clinical reasoning explaining *why* you predict mortality yes/no
- A causal chain using tags:
  <cause>...</cause> → <intermediate effect>...</intermediate effect> → <effect>...</effect>
"""


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 512,
                  temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True) -> str:
    formatted = f"<|User|>{prompt}<|Assistant|>"
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    outputs = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )
    # Strip trailing end markers if present
    for marker in ("<|End|>", "<|end|>", "</s>"):
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()


def extract_causal_chain(text: str) -> Dict[str, Optional[str]]:
    """Parse <cause>, <intermediate effect>, <effect> tags if present."""
    def _one(tag: str) -> Optional[str]:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    return {
        "cause": _one("cause"),
        "intermediate_effect": _one("intermediate effect"),
        "effect": _one("effect"),
    }


def extract_task_explanations(text: str) -> Dict[str, Optional[str]]:
    """
    Pull free-text reasoning near MACE / Mortality prediction blocks.
    """
    lower = text.lower()
    mace_exp = None
    mort_exp = None

    mace_idx = lower.find("mace prediction")
    mort_idx = lower.find("mortality prediction")

    if mace_idx != -1:
        end = mort_idx if mort_idx > mace_idx else len(text)
        block = text[mace_idx:end].strip()
        # Drop the first prediction line; keep following reasoning
        lines = block.splitlines()
        mace_exp = "\n".join(lines[1:]).strip() if len(lines) > 1 else block

    if mort_idx != -1:
        # Stop before causal tags if present
        cause_idx = lower.find("<cause>", mort_idx)
        end = cause_idx if cause_idx != -1 else len(text)
        block = text[mort_idx:end].strip()
        lines = block.splitlines()
        mort_exp = "\n".join(lines[1:]).strip() if len(lines) > 1 else block

    return {"mace_explanation": mace_exp, "mortality_explanation": mort_exp}


def _scalar(value):
    """Convert pandas/numpy scalars to plain Python for JSON."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def run_stage1_for_patient(
    processor: EKGLeadBasedProcessor,
    detector: FrozenAbnormalityDetector,
    patient_data,
    lead_features: Dict[str, List[str]],
    patient_id: Any,
    cache: Optional[Stage1Cache] = None,
) -> Tuple[Dict[str, str], str]:
    """Run frozen Stage I on 12 leads + 2 metadata groups."""
    findings: Dict[str, str] = {}

    for lead in processor.leads:
        feats = lead_features.get(lead) or []
        if not feats:
            continue
        group = f"lead_{lead}"
        if cache is not None:
            hit = cache.get(patient_id, group)
            if hit is not None:
                findings[group] = hit
                continue
        prompt = build_stage1_lead_prompt(processor, patient_data, lead, feats)
        text = detector.detect(prompt)
        if not text.lower().startswith(f"lead {lead.lower()}"):
            text = f"Lead {lead}: {text}"
        findings[group] = text
        if cache is not None:
            cache.set(patient_id, group, text)
            cache.save(verbose=False)

    for group, prompt_fn, prefix in [
        ("metadata_demographics", build_stage1_demo_prompt, "Demographics group"),
        ("metadata_clinical", build_stage1_clinical_prompt, "Clinical metadata group"),
    ]:
        if cache is not None:
            hit = cache.get(patient_id, group)
            if hit is not None:
                findings[group] = hit
                continue
        prompt = prompt_fn(processor, patient_data)
        text = detector.detect(prompt)
        if prefix.lower() not in text.lower()[:60]:
            text = f"{prefix}: {text}"
        findings[group] = text
        if cache is not None:
            cache.set(patient_id, group, text)
            cache.save(verbose=False)

    aggregated = processor._aggregate_group_findings(findings)
    return findings, aggregated


def run_stage2_for_patient(
    model,
    tokenizer,
    processor: EKGLeadBasedProcessor,
    patient_data,
    aggregated_findings: str,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Run fine-tuned Stage II risk prediction with explanations."""
    prompt = build_stage2_prompt(processor, patient_data, aggregated_findings)
    generated = generate_text(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens, do_sample=True
    )

    mace_prob, mort_prob, mace_pred, mort_pred = extract_probabilities_from_text(generated)
    if mace_pred is None:
        mace_pred, mort_pred = extract_predictions_from_text(generated)

    explanations = extract_task_explanations(generated)
    causal = extract_causal_chain(generated)

    return {
        "prompt": prompt,
        "raw_generation": generated,
        "mace_prediction": int(mace_pred),
        "mortality_prediction": int(mort_pred),
        "mace_probability": mace_prob,
        "mortality_probability": mort_prob,
        "mace_explanation": explanations["mace_explanation"],
        "mortality_explanation": explanations["mortality_explanation"],
        "causal_chain": causal,
    }


def _atomic_json_dump(path: str, payload: Any):
    """Write JSON via temp file + replace so crashes don't truncate the file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, path)


def _patient_key(patient_id: Any) -> str:
    if hasattr(patient_id, "item"):
        try:
            return str(patient_id.item())
        except Exception:
            return str(patient_id)
    return str(patient_id)


def infer_patients(
    model,
    tokenizer,
    config: Config,
    csv_path: str,
    num_patients: int,
    stage1_cache_path: Optional[str] = None,
    stage1_results_path: Optional[str] = None,
    stage2_results_path: Optional[str] = None,
    stage2_max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """
    Per-patient pipeline: Stage I (frozen) → Stage II (LoRA) immediately,
    with disk checkpoints after each stage / patient.
    """
    processor = EKGLeadBasedProcessor(config)
    df = processor.load_and_prepare_data(csv_path, max_samples=num_patients)
    lead_features = processor.group_features_by_lead(df)
    cache = Stage1Cache(stage1_cache_path, autosave=False) if stage1_cache_path else None

    stage1_results: Dict[str, Any] = {"patients": {}}
    if stage1_results_path and os.path.exists(stage1_results_path):
        try:
            with open(stage1_results_path, "r") as f:
                stage1_results = json.load(f)
            if "patients" not in stage1_results:
                stage1_results = {"patients": stage1_results}
            print(
                f"Resumed Stage I results: {len(stage1_results['patients'])} patients "
                f"from {stage1_results_path}"
            )
        except Exception as e:
            print(f"Warning: could not resume Stage I results ({e}); starting fresh")

    results: List[Dict[str, Any]] = []
    if stage2_results_path and os.path.exists(stage2_results_path):
        try:
            with open(stage2_results_path, "r") as f:
                prior = json.load(f)
            results = list(prior.get("patients", []))
            print(
                f"Resumed Stage II results: {len(results)} patients "
                f"from {stage2_results_path}"
            )
        except Exception as e:
            print(f"Warning: could not resume Stage II results ({e}); starting fresh")
            results = []

    done_stage2_ids = {_patient_key(r.get("patient_id")) for r in results}

    detector = FrozenAbnormalityDetector(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.model.stage1_max_new_tokens,
        temperature=config.model.stage1_temperature,
        top_p=config.model.top_p,
        do_sample=config.model.stage1_do_sample,
        max_input_length=config.model.stage1_max_input_length,
        auto_disable_adapters=False,  # toggled per patient
    )

    print_section_header("Per-patient Stage I → Stage II")
    for idx, (_, row) in enumerate(df.iterrows()):
        patient_id = row.get("FakeMRN", idx)
        pid_key = _patient_key(patient_id)
        print(f"\n===== Patient {idx + 1}/{len(df)} id={patient_id} =====")

        if pid_key in done_stage2_ids:
            print("  → already complete in Stage II checkpoint; skipping")
            continue

        # ----- Stage I (frozen base) -----
        print(f"[Stage I] Patient {idx + 1}/{len(df)} id={patient_id}")
        prior_s1 = stage1_results["patients"].get(pid_key)
        if prior_s1 and prior_s1.get("group_findings") and prior_s1.get("aggregated_summary"):
            findings = prior_s1["group_findings"]
            aggregated = prior_s1["aggregated_summary"]
            print(f"  → loaded Stage I from disk ({len(findings)} groups)")
        else:
            detector.disable_adapters(verbose=(idx == 0))
            findings, aggregated = run_stage1_for_patient(
                processor, detector, row, lead_features, patient_id, cache=cache
            )
            stage1_results["patients"][pid_key] = {
                "patient_id": _scalar(patient_id) if not isinstance(patient_id, str) else patient_id,
                "group_findings": findings,
                "aggregated_summary": aggregated,
            }
            if stage1_results_path:
                _atomic_json_dump(stage1_results_path, stage1_results)
                print(f"  → checkpointed Stage I → {stage1_results_path}")
            if cache is not None:
                cache.save(verbose=False)

        # ----- Stage II (fine-tuned adapters) immediately -----
        print(f"[Stage II] Patient {idx + 1}/{len(df)} id={patient_id}")
        detector.enable_adapters(verbose=(idx == 0))
        stage2 = run_stage2_for_patient(
            model,
            tokenizer,
            processor,
            row,
            aggregated,
            max_new_tokens=stage2_max_new_tokens,
        )

        true_mace = int(row.get("3p_MACE_binary", 0)) if "3p_MACE_binary" in row.index else None
        true_mort = int(row.get("Mortality_Binary", 0)) if "Mortality_Binary" in row.index else None

        record = {
            "patient_id": _scalar(patient_id) if not isinstance(patient_id, str) else patient_id,
            "demographics": {
                "age": _scalar(row.get("PatientAge")) if "PatientAge" in row.index else None,
                "gender": _scalar(row.get("Gender")) if "Gender" in row.index else None,
                "weight_kg": _scalar(row.get("WeightKg")) if "WeightKg" in row.index else None,
                "height_cm": _scalar(row.get("HeightCm")) if "HeightCm" in row.index else None,
            },
            "stage1": {
                "group_findings": findings,
                "aggregated_summary": aggregated,
                "explanation": (
                    "Stage I (frozen LLM) inspected each lead's measured features and "
                    "two metadata groups; each finding below is the model's abnormality "
                    "analysis and rationale grounded in those inputs."
                ),
            },
            "stage2": {
                "mace_prediction": stage2["mace_prediction"],
                "mortality_prediction": stage2["mortality_prediction"],
                "mace_probability": stage2["mace_probability"],
                "mortality_probability": stage2["mortality_probability"],
                "mace_explanation": stage2["mace_explanation"],
                "mortality_explanation": stage2["mortality_explanation"],
                "causal_chain": stage2["causal_chain"],
                "raw_generation": stage2["raw_generation"],
            },
            "ground_truth": {
                "3p_MACE_binary": true_mace,
                "Mortality_Binary": true_mort,
            },
        }
        results.append(record)
        done_stage2_ids.add(pid_key)

        if stage2_results_path:
            _atomic_json_dump(
                stage2_results_path,
                {
                    "num_patients_completed": len(results),
                    "patients": results,
                },
            )
            print(f"  → checkpointed Stage II ({len(results)}/{len(df)}) → {stage2_results_path}")

        print(
            f"  → MACE={stage2['mace_prediction']} "
            f"(true={true_mace}), Mortality={stage2['mortality_prediction']} "
            f"(true={true_mort})"
        )
        # Stage I findings for this patient are only on disk now — not kept in a big dict

    if cache is not None:
        cache.save(verbose=True)
    # Leave adapters enabled for any follow-on use
    detector.enable_adapters(verbose=False)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CLAIRE-α inference with Stage I + Stage II explanations"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=_DEFAULT_MODEL,
        help="Path to fine-tuned LoRA checkpoint directory",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=_DEFAULT_CSV,
        help="Path to EKG CSV (same schema as training)",
    )
    parser.add_argument(
        "--num_patients",
        type=int,
        default=20,
        help="Number of patients to run (default: 20)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs",
            "inference",
        ),
        help="Directory for JSON outputs (default: CLAIRE-Alpha/outputs/inference)",
    )
    parser.add_argument(
        "--stage1_cache",
        type=str,
        default=None,
        help="Optional Stage I findings cache JSON",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = _resolve_path(args.model_path)
    csv_path = _resolve_path(args.csv_path)
    output_dir = _resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Timestamped JSON filename under outputs/inference/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"inference_results_{timestamp}.json"

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")

    config = Config(
        model=ModelConfig(),
        data=DataConfig(csv_path=csv_path, max_samples=args.num_patients),
        training=TrainingConfig(output_dir=output_dir),
    )
    set_seed(args.seed)

    print("=" * 80)
    print("CLAIRE-α Inference")
    print("=" * 80)
    print(f"Model:         {model_path}")
    print(f"Data:          {csv_path}")
    print(f"Base weights:  {config.model.model_name}")
    print(f"Patients:      {args.num_patients}")
    print(f"Output:        {output_dir}")
    print(f"CUDA:          {torch.cuda.is_available()}")
    print("=" * 80)

    model, tokenizer, _ = load_finetuned_model(model_path, config)

    cache_path = args.stage1_cache
    if cache_path is None:
        cache_path = os.path.join(output_dir, "stage1_cache.json")
    else:
        cache_path = _resolve_path(cache_path)

    stage1_results_path = os.path.join(output_dir, "stage1_results.json")
    stage2_results_path = os.path.join(output_dir, "stage2_results.json")
    print(f"Stage I checkpoints → {stage1_results_path}")
    print(f"Stage II checkpoints → {stage2_results_path}")

    results = infer_patients(
        model=model,
        tokenizer=tokenizer,
        config=config,
        csv_path=csv_path,
        num_patients=args.num_patients,
        stage1_cache_path=cache_path,
        stage1_results_path=stage1_results_path,
        stage2_results_path=stage2_results_path,
    )

    meta = {
        "timestamp": timestamp,
        "model_path": model_path,
        "csv_path": csv_path,
        "base_model": config.model.model_name,
        "num_patients": len(results),
        "seed": args.seed,
        "stage1_results_path": stage1_results_path,
        "stage2_results_path": stage2_results_path,
        "stage1_cache_path": cache_path,
    }
    final_payload = {"meta": meta, "patients": results}
    # Write primary JSON to outputs/inference/
    json_path = os.path.join(output_dir, json_filename)
    _atomic_json_dump(json_path, final_payload)
    latest_path = os.path.join(output_dir, "inference_results.json")
    _atomic_json_dump(latest_path, final_payload)
    print(f"Saved JSON: {json_path}")
    print(f"Saved JSON (latest): {latest_path}")
    print("\nInference complete.")


if __name__ == "__main__":
    main()
