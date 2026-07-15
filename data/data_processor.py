"""
EKG Data Processor for Lead-Based Fine-tuning

Stage I uses a frozen LLM to analyze real patient EKG features
(12 leads + 2 metadata groups). Stage II builds risk-prediction examples
from those LLM-detected abnormalities.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split


class EKGLeadBasedProcessor:
    """
    Data processor for lead-based EKG analysis training.
    """

    def __init__(self, config):
        self.config = config
        self.leads = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVR', 'aVL', 'aVF']
        self.demographic_columns = [
            'PatientAge', 'Gender', 'Race', 'Ethnicity', 'WeightKg', 'HeightCm'
        ]
        self.clinical_metadata_columns = [
            'TestReason', 'HISDisposition', 'DATE_DIFF_days_Acq_Edit'
        ]

    def load_and_prepare_data(self, csv_path: str, max_samples: int = 5000) -> pd.DataFrame:
        """Load and prepare EKG data"""
        print(f"Loading EKG data from {csv_path}")

        df = pd.read_csv(csv_path)

        if max_samples and len(df) > max_samples:
            df = df.head(max_samples)
            print(f"Limited dataset to {max_samples} samples")

        print(f"Loaded {len(df)} samples with {len(df.columns)} features")

        df = self._clean_data(df)
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        for col in self.demographic_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())

        for col in self.clinical_metadata_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')

        for target_col in self.config.data.target_columns:
            if target_col in df.columns:
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)

        return df

    def group_features_by_lead(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Group features by EKG lead"""
        lead_features = {}
        for lead in self.leads:
            lead_columns = [col for col in df.columns if col.endswith(f'_{lead}')]
            lead_features[lead] = lead_columns
        return lead_features

    def format_lead_data(self, patient_data: pd.Series, lead: str, lead_features: List[str]) -> str:
        """Format lead data for Stage I model input"""
        demographics = self._format_demographics(patient_data)
        lead_measurements = self._format_lead_measurements(patient_data, lead_features)

        return f"""Patient Information:
{demographics}

EKG Lead {lead} Analysis:
{lead_measurements}

Please analyze this EKG lead {lead} measurement data and identify any abnormal patterns or concerning findings.
Focus on clinically significant abnormalities that could indicate cardiac pathology.
Base your assessment only on the provided measurements. If findings appear within normal limits, state that explicitly.
"""

    def format_demographics_group_prompt(self, patient_data: pd.Series) -> str:
        """Stage I metadata group 1: demographics"""
        demographics = self._format_demographics(patient_data)
        return f"""Patient Demographic Metadata:
{demographics}

Please assess demographic cardiovascular risk context from this metadata group.
Comment on age, sex, and body-size related risk considerations relevant to EKG interpretation.
Do not invent lab values or diagnoses not supported by the provided metadata.
"""

    def format_clinical_metadata_group_prompt(self, patient_data: pd.Series) -> str:
        """Stage I metadata group 2: clinical / acquisition metadata"""
        lines = []
        for col in self.clinical_metadata_columns:
            if col in patient_data.index and pd.notna(patient_data.get(col)):
                lines.append(f"- {col}: {patient_data[col]}")
        if not lines:
            lines.append("- No clinical metadata fields available")

        metadata_block = "\n".join(lines)
        return f"""Clinical / Acquisition Metadata:
{metadata_block}

Please assess clinically relevant context from this metadata group for cardiovascular risk and EKG interpretation
(e.g., test indication, disposition, timing). Do not invent facts beyond the provided fields.
"""

    def _format_demographics(self, patient_data: pd.Series) -> str:
        """Format patient demographics"""
        age = patient_data.get('PatientAge', 'Unknown')
        gender = patient_data.get('Gender', 'Unknown')
        if gender == 'M':
            gender = 'Male'
        elif gender == 'F':
            gender = 'Female'

        weight = patient_data.get('WeightKg', 'Unknown')
        height = patient_data.get('HeightCm', 'Unknown')

        race = patient_data.get('Race', 'Unknown') if 'Race' in patient_data.index else 'Unknown'
        ethnicity = patient_data.get('Ethnicity', 'Unknown') if 'Ethnicity' in patient_data.index else 'Unknown'

        return f"""- Age: {age} years
- Gender: {gender}
- Race: {race}
- Ethnicity: {ethnicity}
- Weight: {weight} kg
- Height: {height} cm"""

    def _format_lead_measurements(self, patient_data: pd.Series, lead_features: List[str]) -> str:
        """Format lead measurements by category"""
        measurements: Dict[str, List[str]] = {}

        for feature in lead_features:
            if pd.notna(patient_data.get(feature)):
                category = self._categorize_measurement(feature)
                measurements.setdefault(category, []).append(
                    f"  - {feature}: {patient_data[feature]}"
                )

        formatted_text = ""
        for category, measures in measurements.items():
            if measures:
                formatted_text += f"\n{category}:\n" + "\n".join(measures) + "\n"
        return formatted_text

    def _categorize_measurement(self, feature_name: str) -> str:
        """Categorize measurement by feature name"""
        feature_lower = feature_name.lower()

        if any(x in feature_lower for x in ['p_', 'pp_']):
            return "P Wave Measurements"
        elif any(x in feature_lower for x in ['qrs', 'q_', 'r_', 's_', 'rp_', 'sp_']):
            return "QRS Complex Measurements"
        elif any(x in feature_lower for x in ['t_', 'tp_', 'tfull']):
            return "T Wave Measurements"
        elif any(x in feature_lower for x in ['st', 'ste_', 'stj_', 'stm_']):
            return "ST Segment Measurements"
        else:
            return "General Measurements"

    def _detect_group_abnormality(
        self,
        detector,
        cache,
        patient_id: Any,
        group: str,
        prompt: str,
    ) -> str:
        """Run or look up Stage I abnormality detection for one feature group."""
        if cache is not None:
            cached = cache.get(patient_id, group)
            if cached is not None:
                return cached

        if detector is None:
            raise ValueError(
                "Stage I requires a FrozenAbnormalityDetector (frozen LLM). "
                "Simulation of abnormalities is no longer supported."
            )

        finding = detector.detect(prompt)
        # Keep lead/group attribution clear for aggregation
        if group.startswith("lead_"):
            lead = group.replace("lead_", "", 1)
            if not finding.lower().startswith(f"lead {lead.lower()}"):
                finding = f"Lead {lead}: {finding}"
        elif group == "metadata_demographics":
            if "demographic" not in finding.lower()[:40]:
                finding = f"Demographics group: {finding}"
        elif group == "metadata_clinical":
            if "clinical" not in finding.lower()[:40] and "metadata" not in finding.lower()[:40]:
                finding = f"Clinical metadata group: {finding}"

        if cache is not None:
            cache.set(patient_id, group, finding)
        return finding

    def create_training_examples(
        self,
        df: pd.DataFrame,
        detector=None,
        cache=None,
    ) -> List[Dict[str, Any]]:
        """
        Create training examples using Stage I frozen-LLM abnormality detection
        on 12 leads + 2 metadata groups, then Stage II risk-prediction targets.
        """
        if detector is None and cache is None:
            raise ValueError(
                "create_training_examples requires a Stage I FrozenAbnormalityDetector "
                "and/or a populated Stage1Cache. Abnormalities are no longer simulated."
            )

        lead_features = self.group_features_by_lead(df)
        training_examples = []

        print(f"Creating training examples for {len(df)} patients (Stage I: frozen LLM)...")

        for idx, (_, patient_data) in enumerate(df.iterrows()):
            if idx % 10 == 0:
                print(f"Processing patient {idx + 1}/{len(df)}")

            patient_id = patient_data.get('FakeMRN', idx)
            group_findings: Dict[str, str] = {}

            # --- Stage I: 12 leads ---
            for lead in self.leads:
                if lead in lead_features and lead_features[lead]:
                    lead_prompt = self.format_lead_data(
                        patient_data, lead, lead_features[lead]
                    )
                    finding = self._detect_group_abnormality(
                        detector, cache, patient_id, f"lead_{lead}", lead_prompt
                    )
                    group_findings[f"lead_{lead}"] = finding

                    training_examples.append({
                        'type': 'lead_analysis',
                        'patient_id': patient_id,
                        'lead': lead,
                        'text': f"<|User|>{lead_prompt}<|Assistant|>{finding}<|End|>",
                        'mace_label': int(patient_data.get('3p_MACE_binary', 0)),
                        'mortality_label': int(patient_data.get('Mortality_Binary', 0)),
                    })

            # --- Stage I: 2 metadata groups ---
            demo_prompt = self.format_demographics_group_prompt(patient_data)
            demo_finding = self._detect_group_abnormality(
                detector, cache, patient_id, "metadata_demographics", demo_prompt
            )
            group_findings["metadata_demographics"] = demo_finding
            training_examples.append({
                'type': 'metadata_analysis',
                'patient_id': patient_id,
                'lead': 'demographics',
                'text': f"<|User|>{demo_prompt}<|Assistant|>{demo_finding}<|End|>",
                'mace_label': int(patient_data.get('3p_MACE_binary', 0)),
                'mortality_label': int(patient_data.get('Mortality_Binary', 0)),
            })

            clinical_prompt = self.format_clinical_metadata_group_prompt(patient_data)
            clinical_finding = self._detect_group_abnormality(
                detector, cache, patient_id, "metadata_clinical", clinical_prompt
            )
            group_findings["metadata_clinical"] = clinical_finding
            training_examples.append({
                'type': 'metadata_analysis',
                'patient_id': patient_id,
                'lead': 'clinical_metadata',
                'text': f"<|User|>{clinical_prompt}<|Assistant|>{clinical_finding}<|End|>",
                'mace_label': int(patient_data.get('3p_MACE_binary', 0)),
                'mortality_label': int(patient_data.get('Mortality_Binary', 0)),
            })

            # --- Stage II example: risk prediction from Stage I findings ---
            aggregated = self._aggregate_group_findings(group_findings)
            risk_prompt = self._create_risk_prediction_prompt(patient_data, aggregated)
            risk_response = self._create_risk_prediction_response(patient_data)

            training_examples.append({
                'type': 'risk_prediction',
                'patient_id': patient_id,
                'lead': 'all',
                'text': f"<|User|>{risk_prompt}<|Assistant|>{risk_response}<|End|>",
                'mace_label': int(patient_data.get('3p_MACE_binary', 0)),
                'mortality_label': int(patient_data.get('Mortality_Binary', 0)),
            })

        if cache is not None:
            cache.save()

        print(f"Created {len(training_examples)} training examples")
        print(
            f"Lead analysis: {sum(1 for ex in training_examples if ex['type'] == 'lead_analysis')}"
        )
        print(
            f"Metadata analysis: {sum(1 for ex in training_examples if ex['type'] == 'metadata_analysis')}"
        )
        print(
            f"Risk prediction: {sum(1 for ex in training_examples if ex['type'] == 'risk_prediction')}"
        )
        return training_examples

    def _aggregate_group_findings(self, group_findings: Dict[str, str]) -> str:
        """Aggregate Stage I findings from leads + metadata groups"""
        significant_findings = []

        for group, finding in group_findings.items():
            finding_lower = finding.lower()
            if any(
                phrase in finding_lower
                for phrase in [
                    'no significant',
                    'within normal',
                    'appears normal',
                    'normal limits',
                    'unremarkable',
                ]
            ) and not any(
                phrase in finding_lower
                for phrase in ['abnormal', 'concerning', 'elevated risk', 'patholog']
            ):
                continue
            significant_findings.append(finding)

        if not significant_findings:
            return "No significant abnormalities detected across EKG leads and metadata groups."

        return "EKG Abnormality Summary (Stage I frozen LLM):\n\n" + "\n\n".join(
            significant_findings
        )

    def _create_risk_prediction_prompt(self, patient_data: pd.Series, abnormalities: str) -> str:
        """Create prompt for Stage II risk prediction"""
        demographics = self._format_demographics(patient_data)

        return f"""Patient Demographics:
{demographics}

{abnormalities}

Based on the patient demographics and EKG abnormalities identified above, please assess the risk for:
1. Major Adverse Cardiac Events (MACE) within 3 years
2. Mortality risk

Provide binary predictions (0 = No, 1 = Yes), estimated probabilities (0.0-1.0), and clinical reasoning.
"""

    def _create_risk_prediction_response(self, patient_data: pd.Series) -> str:
        """Create supervised response for Stage II risk prediction"""
        mace_outcome = int(patient_data.get('3p_MACE_binary', 0))
        mortality_outcome = int(patient_data.get('Mortality_Binary', 0))
        mace_prob = float(mace_outcome)
        mortality_prob = float(mortality_outcome)

        if mace_outcome == 1:
            mace_reasoning = "The combination of EKG abnormalities and patient factors suggests elevated cardiovascular risk."
        else:
            mace_reasoning = "Despite some findings, the overall risk profile appears manageable with standard care."

        if mortality_outcome == 1:
            mortality_reasoning = "The identified abnormalities combined with patient characteristics indicate increased mortality risk."
        else:
            mortality_reasoning = "Current findings do not suggest significantly elevated mortality risk."

        return f"""Based on the clinical assessment:

1. **MACE Prediction: {mace_outcome}** (probability: {mace_prob:.1f}) (0 = No, 1 = Yes)
{mace_reasoning}

2. **Mortality Prediction: {mortality_outcome}** (probability: {mortality_prob:.1f}) (0 = No, 1 = Yes)
{mortality_reasoning}

<cause>Patient demographics and EKG findings</cause> → <intermediate effect>Cardiovascular risk stratification</intermediate effect> → <effect>Clinical risk predictions for MACE and mortality</effect>
"""

    def split_data(self, training_examples: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """Split data into train/eval/test sets at patient level"""
        patient_examples: Dict[Any, List] = {}
        for example in training_examples:
            patient_id = example['patient_id']
            patient_examples.setdefault(patient_id, []).append(example)

        patient_ids = list(patient_examples.keys())

        train_patients, temp_patients = train_test_split(
            patient_ids,
            test_size=(1 - self.config.data.train_split),
            random_state=self.config.seed,
            stratify=None,
        )

        eval_size = self.config.data.eval_split / (
            self.config.data.eval_split + self.config.data.test_split
        )
        eval_patients, test_patients = train_test_split(
            temp_patients,
            test_size=(1 - eval_size),
            random_state=self.config.seed,
        )

        train_examples = []
        eval_examples = []
        test_examples = []

        for patient_id in train_patients:
            train_examples.extend(patient_examples[patient_id])
        for patient_id in eval_patients:
            eval_examples.extend(patient_examples[patient_id])
        for patient_id in test_patients:
            test_examples.extend(patient_examples[patient_id])

        print(
            f"Data split - Train: {len(train_examples)}, "
            f"Eval: {len(eval_examples)}, Test: {len(test_examples)}"
        )
        return train_examples, eval_examples, test_examples
