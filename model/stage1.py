"""
Stage I: Frozen LLM abnormality detection from structured EKG features.

Matches the CLAIRE-α README pipeline: a frozen pre-trained DeepSeek model is
prompted on group-wise EKG features (12 leads + 2 metadata groups) to produce
patient-level abnormality findings used by Stage II risk prediction.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Any

import torch


class FrozenAbnormalityDetector:
    """
    Run frozen (no-gradient) causal-LM inference for Stage I abnormality detection.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = False,
        max_input_length: int = 8192,
        auto_disable_adapters: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.max_input_length = max_input_length
        self.model.eval()
        self._adapters_disabled = False

        if auto_disable_adapters:
            self.disable_adapters(verbose=True)

    def disable_adapters(self, verbose: bool = False):
        """Turn off PEFT adapters so Stage I uses the frozen base model."""
        if self._adapters_disabled:
            return
        if hasattr(self.model, "disable_adapter_layers"):
            try:
                self.model.disable_adapter_layers()
                self._adapters_disabled = True
                if verbose:
                    print("Stage I: PEFT adapters disabled (using frozen base model)")
            except Exception as e:
                if verbose:
                    print(f"Warning: could not disable adapters: {e}")

    def enable_adapters(self, verbose: bool = False):
        """Re-enable PEFT adapters for Stage II fine-tuned inference."""
        if not self._adapters_disabled:
            return
        if hasattr(self.model, "enable_adapter_layers"):
            try:
                self.model.enable_adapter_layers()
                self._adapters_disabled = False
                if verbose:
                    print("Stage II: PEFT adapters re-enabled (using fine-tuned model)")
            except Exception as e:
                if verbose:
                    print(f"Warning: could not enable adapters: {e}")

    def restore_adapters(self):
        """Alias for enable_adapters (backward compatible)."""
        self.enable_adapters(verbose=True)

    @torch.inference_mode()
    def detect(self, prompt: str) -> str:
        """
        Prompt the frozen LLM with structured EKG group features and return
        its abnormality analysis text.
        """
        formatted = f"<|User|>{prompt}<|Assistant|>"
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        outputs = self.model.generate(**inputs, **generate_kwargs)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()


class Stage1Cache:
    """Disk cache for Stage I findings keyed by patient_id + group name."""

    def __init__(self, cache_path: Optional[str] = None, autosave: bool = False):
        self.cache_path = cache_path
        self.autosave = autosave
        self._store: Dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self._store = json.load(f)
            print(f"Loaded Stage I cache with {len(self._store)} entries from {cache_path}")

    @staticmethod
    def make_key(patient_id: Any, group: str) -> str:
        return f"{patient_id}::{group}"

    def get(self, patient_id: Any, group: str) -> Optional[str]:
        return self._store.get(self.make_key(patient_id, group))

    def set(self, patient_id: Any, group: str, finding: str):
        self._store[self.make_key(patient_id, group)] = finding
        if self.autosave:
            self.save(verbose=False)

    def save(self, verbose: bool = True):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._store, f, indent=2)
        os.replace(tmp_path, self.cache_path)
        if verbose:
            print(f"Saved Stage I cache ({len(self._store)} entries) to {self.cache_path}")
