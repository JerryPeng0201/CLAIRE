#!/usr/bin/env python3
"""
Convert Stage II raw_generation texts into compact causal chains using the
local DeepSeek LLM. Writes a NEW JSON file; does not modify source inference JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_raw_generation(text: str) -> str:
    """Remove truncated gibberish / special tokens while keeping clinical content."""
    if not text:
        return ""
    cut_markers = [
        "<｜end▁of▁sentence｜>",
        "<|end",
        "<｜User｜>",
        "<|User|>",
        "**End of thinking",
        "**End of assessment",
        "**End of thought",
        "**End of summary",
        "**End of clinical",
    ]
    cleaned = text
    for marker in cut_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]
    # Drop lines dominated by non-ASCII corruption
    keep = []
    for line in cleaned.splitlines():
        bad = sum(1 for c in line if ord(c) > 127)
        if bad > 8:
            continue
        keep.append(line)
    return "\n".join(keep).strip()


PROMPT_TEMPLATE = """You convert Stage-II EKG risk assessment text into SHORT causal chains.

Output STRICT JSON only (no markdown, no extra keys):
{{
  "mace_chain": "reason1 -> intermediate1 -> intermediate2 -> MACE",
  "mortality_chain": "reason1 -> intermediate1 -> intermediate2 -> Mortality",
  "combined_chain": "reason1 -> intermediate1 -> intermediate2 -> MACE/Mortality"
}}

Rules:
- Use the format: reason -> result/reason -> result/reason -> MACE and/or Mortality
- Extract concrete EKG/clinical reasons from the text when present (e.g. ST elevation, LBBB, LVH, advanced age).
- If the text is vague, use the stated MACE/Mortality predictions and brief generic clinical reasons.
- Each chain should have 3-5 nodes separated by " -> ".
- Final node must end with MACE, Mortality, or MACE/Mortality as appropriate.
- Predictions: MACE={mace_pred}, Mortality={mort_pred}.
- Prefer positive findings when prediction=1; protective/low-risk reasons when prediction=0.

Stage II raw generation:
\"\"\"
{raw_text}
\"\"\"
"""


def build_fallback_chains(raw: str, mace_pred: int, mort_pred: int) -> Dict[str, str]:
    """Deterministic backup if LLM JSON parse fails."""
    findings = []
    patterns = [
        (r"(?i)st elevation", "ST elevation"),
        (r"(?i)st depression", "ST depression"),
        (r"(?i)t wave inversion", "T-wave inversion"),
        (r"(?i)bundle branch block|lbbb|rbbb", "bundle branch block"),
        (r"(?i)left ventricular hypertrophy|lvh", "LVH"),
        (r"(?i)right ventricular hypertrophy", "RVH"),
        (r"(?i)atrial (fibrillation|flutter|enlargement)", "atrial abnormality"),
        (r"(?i)q wave", "Q waves / prior infarct pattern"),
        (r"(?i)ischemia", "ischemic changes"),
        (r"(?i)hypertrophy", "ventricular hypertrophy"),
    ]
    for pat, label in patterns:
        if re.search(pat, raw or "") and label not in findings:
            findings.append(label)
        if len(findings) >= 2:
            break
    if not findings:
        findings = ["EKG abnormalities and patient risk factors"]

    r1 = findings[0]
    r2 = findings[1] if len(findings) > 1 else "cardiovascular risk stratification"

    if mace_pred == 1:
        mace_chain = f"{r1} -> {r2} -> elevated CV risk -> MACE"
    else:
        mace_chain = f"{r1} -> manageable risk profile -> standard care sufficient -> no MACE"

    if mort_pred == 1:
        mort_chain = f"{r1} -> {r2} -> adverse prognosis -> Mortality"
    else:
        mort_chain = f"{r1} -> limited mortality signals -> preserved outlook -> no Mortality"

    if mace_pred == 1 and mort_pred == 1:
        combined = f"{r1} -> {r2} -> elevated risk stratification -> MACE/Mortality"
    elif mace_pred == 1:
        combined = f"{r1} -> {r2} -> event risk without clear mortality surge -> MACE"
    elif mort_pred == 1:
        combined = f"{r1} -> {r2} -> mortality-dominant risk -> Mortality"
    else:
        combined = f"{r1} -> stable clinical profile -> low predicted risk -> no MACE/Mortality"

    return {
        "mace_chain": mace_chain,
        "mortality_chain": mort_chain,
        "combined_chain": combined,
    }


def extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    # strip fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def load_local_llm(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading converter LLM from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=os.path.isdir(model_path)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": os.path.isdir(model_path),
    }
    # Prefer 4-bit if CUDA + bitsandbytes available
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def llm_convert(model, tokenizer, raw: str, mace_pred: int, mort_pred: int) -> Dict[str, str]:
    import torch

    prompt = PROMPT_TEMPLATE.format(
        mace_pred=mace_pred, mort_pred=mort_pred, raw_text=raw[:3500]
    )
    # Prefer chat-like markers used elsewhere in this codebase when helpful
    formatted = f"<|User|>{prompt}<|Assistant|>"
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=4096
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    parsed = extract_json_object(gen)
    if not parsed:
        return build_fallback_chains(raw, mace_pred, mort_pred)

    # normalize keys
    mace_chain = parsed.get("mace_chain") or parsed.get("MACE_chain")
    mort_chain = parsed.get("mortality_chain") or parsed.get("Mortality_chain")
    combined = parsed.get("combined_chain") or parsed.get("combined")
    if not (mace_chain and mort_chain and combined):
        fb = build_fallback_chains(raw, mace_pred, mort_pred)
        mace_chain = mace_chain or fb["mace_chain"]
        mort_chain = mort_chain or fb["mortality_chain"]
        combined = combined or fb["combined_chain"]
    return {
        "mace_chain": str(mace_chain).strip(),
        "mortality_chain": str(mort_chain).strip(),
        "combined_chain": str(combined).strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/inference/inference_results_20260715_093849.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: outputs/inference/stage2_causal_chains_<timestamp>.json",
    )
    parser.add_argument(
        "--model_path",
        default="model/weights/DeepSeek-R1-0528-Qwen3-8B",
    )
    parser.add_argument(
        "--fallback_only",
        action="store_true",
        help="Skip LLM load; use rule-based conversion only",
    )
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = args.input if os.path.isabs(args.input) else os.path.join(repo, args.input)
    model_path = (
        args.model_path
        if os.path.isabs(args.model_path)
        else os.path.join(repo, args.model_path)
    )

    with open(in_path) as f:
        data = json.load(f)

    patients = data.get("patients", [])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(in_path), f"stage2_causal_chains_{timestamp}.json"
        )
    elif not os.path.isabs(out_path):
        out_path = os.path.join(repo, out_path)

    model = tokenizer = None
    converter = "rule_based_fallback"
    if not args.fallback_only and os.path.isdir(model_path):
        try:
            model, tokenizer = load_local_llm(model_path)
            converter = "DeepSeek-R1-0528-Qwen3-8B"
        except Exception as e:
            print(f"LLM load failed ({e}); using rule-based fallback", flush=True)
            converter = "rule_based_fallback"

    results: List[Dict[str, Any]] = []
    for i, pt in enumerate(patients):
        s2 = pt.get("stage2", {})
        raw = clean_raw_generation(s2.get("raw_generation") or "")
        mace = int(s2.get("mace_prediction", 0) or 0)
        mort = int(s2.get("mortality_prediction", 0) or 0)

        if model is not None:
            chains = llm_convert(model, tokenizer, raw, mace, mort)
            method = converter
        else:
            chains = build_fallback_chains(raw, mace, mort)
            method = "rule_based_fallback"

        results.append(
            {
                "patient_id": pt.get("patient_id"),
                "mace_prediction": mace,
                "mortality_prediction": mort,
                "mace_chain": chains["mace_chain"],
                "mortality_chain": chains["mortality_chain"],
                "combined_chain": chains["combined_chain"],
                "conversion_method": method,
                "source_raw_generation_chars": len(raw),
            }
        )
        print(f"Converted patient {i+1}/{len(patients)}", flush=True)

        # checkpoint after each patient
        payload = {
            "meta": {
                "source_file": in_path,
                "created_at": timestamp,
                "converter_model": converter,
                "num_patients": len(results),
                "note": "Derived Stage-II causal chains; original inference JSON was not modified.",
            },
            "patients": results,
        }
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, out_path)

    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
