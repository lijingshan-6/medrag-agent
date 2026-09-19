"""Freeze the manually curated VeritasMed benchmark.

The local models created and challenged the candidate pool.  This file contains
the human curator's explicit selection and source-based rewrites.  It never
consults retrieval rank or model score when choosing the final questions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medrag.benchmark.inventory import build_normalized_chunks, sha256_file
from medrag.benchmark.schema import BenchmarkQuestion, ReviewEvent
from medrag.benchmark.review import (
    append_review_event,
    current_status,
    load_review_events,
)
from medrag.benchmark.selection import freeze_questions
from medrag.benchmark.validation import validate_questions

SELECTED_IDS = [
    "VMG-001", "VMG-002", "VMG-003", "VMG-004", "VMG-006",
    "VMG-007", "VMG-008", "VMG-010", "VMG-011", "VMG-012",
    "VMG-013", "VMG-014", "VMG-015", "VMG-016", "VMG-017",
    "VMG-018", "VMG-019", "VMG-021", "VMG-025", "VMG-026",
    "VMG-027", "VMG-029", "VMG-031", "VMG-035", "VMG-036",
    "VMG-038", "VMG-040", "VMG-041", "VMG-042", "VMG-043",
    "VMG-044", "VMG-046", "VMG-049", "VMG-051", "VMG-052",
    "VMG-054", "VMG-055", "VMG-058", "VMG-059", "VMG-060", "VMG-062",
    "VMG-064", "VMG-065", "VMG-067", "VMG-070", "VMG-074",
    "VMG-075", "VMG-077", "VMG-079", "VMG-080",
]

DEVELOPMENT_IDS = {
    "VMG-001", "VMG-002", "VMG-004", "VMG-006", "VMG-007",
    "VMG-008", "VMG-010", "VMG-013", "VMG-015", "VMG-016",
    "VMG-017", "VMG-018", "VMG-029", "VMG-054", "VMG-058",
}

REJECTION_REASONS = {
    "VMG-005": "Overlaps VMG-004 on the same two atrial-strain studies.",
    "VMG-009": "The prostate-AI source already supports cross-document, limitation, and abstention items; this baseline-only variant was removed to reduce source repetition.",
    "VMG-020": "Duplicates the neuroimaging pair retained in VMG-021 and overstates one method description.",
    "VMG-022": "Infers unmeasured confounding from a sample-size sentence rather than source evidence.",
    "VMG-023": "Asks for outcomes but cites only cohort enrollment.",
    "VMG-024": "The multiple-sclerosis comparison is unrelated to its neuroimaging passages.",
    "VMG-028": "One claimed comparison lacks an outcome passage and VMG-029 covers the source pair better.",
    "VMG-030": "Adds causality and generalizability limitations without grounding them in study design.",
    "VMG-032": "Generic hypothyroidism abstention is unrelated to the selected thyroid-nodule passages.",
    "VMG-033": "Exact duplicate of VMG-017 under the wrong domain.",
    "VMG-034": "Duplicates the HIV cardiac-fibrosis source used in VMG-001 and VMG-007.",
    "VMG-037": "Weaker, less direct version of the AF procedural comparison in VMG-036.",
    "VMG-039": "Duplicates the sTBI cohort sentence without answering the stated outcome question.",
    "VMG-045": "The first evidence span does not support the claimed imaging-method comparison.",
    "VMG-047": "Preclinical retinal evidence is mislabeled as nephrology and adds a generic missing-outcome clause.",
    "VMG-048": "The dialysis-modality question is too generic and not tied to a named comparison.",
    "VMG-050": "Duplicates the ZJP source and was assigned to the wrong clinical domain.",
    "VMG-053": "Both claims combine mechanisms that are absent from their attached evidence spans.",
    "VMG-056": "The inflammatory-bowel-disease comparison is unrelated to the HCC source passages.",
    "VMG-057": "Exact duplicate of VMG-017 under the wrong domain.",
    "VMG-061": "Overlaps the two HFpEF studies retained in the more specific VMG-060.",
    "VMG-063": "Duplicates the sTBI cohort and is mislabeled as pulmonology.",
    "VMG-066": "Duplicates the SCAR-Net study already represented in VMG-011 and cross-document VMG-013.",
    "VMG-068": "Duplicate of the neuroimaging comparison and mislabeled as women's health.",
    "VMG-069": "Duplicate of the neuroimaging comparison and mislabeled as women's health.",
    "VMG-071": "CTA evidence is mislabeled as women's health and the missing-evidence clause is generic.",
    "VMG-072": "The postpartum-depression question is unrelated to its imaging passages.",
    "VMG-073": "The lumbar-multifidus source is retained for a limitation and a matched abstention case; this simpler variant was removed to reduce repetition.",
    "VMG-076": "Exact source-pair duplicate of VMG-012 under the wrong domain.",
    "VMG-078": "Food-protein preservation is outside the medical-literature scope of this benchmark.",
}

STUDY_DESIGNS = {
    "41791688": "cross-sectional pilot study",
    "41960994": "retrospective multicenter simulation study",
    "42065801": "multicenter retrospective cohort study",
    "41477624": "systematic review and meta-analysis",
    "41967781": "preclinical cell and mouse study",
    "42041167": "retrospective single-center feasibility study",
    "42047408": "retrospective single-center model development and validation study",
    "42062228": "prospective randomized imaging-protocol study",
    "41927242": "secondary data analysis",
    "41765253": "prospective single-arm observational cohort",
    "42054004": "cross-sectional study",
    "42023157": "multicenter diagnostic model validation study",
    "41614571": "cross-sectional case-control neuroimaging study",
    "42057682": "retrospective analytical cross-sectional study",
    "41973512": "small observational cohort",
    "42036403": "retrospective single-center observational study",
    "41492270": "retrospective imaging-model development study",
    "42058324": "prospective single-arm radiotherapy study",
    "41698552": "retrospective multicenter observational study",
    "41653331": "retrospective diagnostic accuracy study",
    "41951145": "technical validation study",
    "41903663": "technical validation study",
    "42003534": "retrospective single-center cohort",
    "41860416": "small retrospective comparative study",
    "42065181": "multicenter retrospective cohort study",
    "42041074": "retrospective administrative trend analysis",
    "41657050": "single-cohort prediction-model development study",
    "42044518": "retrospective seven-patient imaging case series",
    "41812546": "preclinical in vitro and animal study",
    "41653578": "preclinical in vitro and animal study",
    "41933596": "small observational exercise study",
    "41864586": "retrospective single-center prognostic study",
    "42069508": "multicenter retrospective administrative cohort",
    "41631491": "diagnostic accuracy study",
    "42077124": "observational diagnostic prediction study",
    "41871491": "observational obstetric cohort",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _claim(
    claim_id: str,
    text: str,
    *evidence: tuple[str, str, str, str | None],
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "text": text,
        "importance": "required",
        "evidence": [
            {
                "chunk_id": chunk_id,
                "quote": quote,
                "support": support,
                "derivation": derivation,
            }
            for chunk_id, quote, support, derivation in evidence
        ],
    }


def _source_index(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for source in candidate["sources"]:
            if source.get("pmid"):
                sources[source["pmid"]] = deepcopy(source)
    return sources


def _sources(
    source_index: dict[str, dict[str, Any]],
    *pmids: str,
    role: str = "primary_study",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pmid in pmids:
        source = deepcopy(source_index[pmid])
        source["role"] = role
        source["study_design"] = STUDY_DESIGNS.get(pmid, source["study_design"])
        rows.append(source)
    return rows


def _rubric(row: dict[str, Any]) -> dict[str, list[str]]:
    if row["answerability"] == "unanswerable":
        return {
            "complete_if": [
                "States that the supplied evidence cannot support the requested comparison or recommendation and names the missing evidence."
            ],
            "partial_if": [
                "Expresses uncertainty but does not identify the missing comparator, outcome, or study design."
            ],
            "fail_if": [
                "Chooses an intervention, reports a comparative effect, or makes a clinical recommendation from the hard-negative passages."
            ],
        }
    required = ", ".join(claim["claim_id"] for claim in row["gold_claims"])
    complete = f"All required claims ({required}) are present with the study population, comparison, numbers, and uncertainty qualifiers preserved."
    if row["answerability"] == "partial":
        complete += " The answer also states the declared evidence gap without filling it by inference."
    return {
        "complete_if": [complete],
        "partial_if": [
            "The main direction is correct but a required number, comparator, population, or evidence boundary is omitted."
        ],
        "fail_if": [
            "A required claim is contradicted, an association is made causal, or an unsupported clinical recommendation is introduced."
        ],
    }


def _finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    row["review"] = {
        "evidence_checked": True,
        "ambiguity_checked": True,
        "answer_checked": True,
        "status": "frozen",
    }
    row["rubric"] = _rubric(row)
    row["supporting_chunk_ids"] = sorted(
        {
            evidence["chunk_id"]
            for claim in row["gold_claims"]
            for evidence in claim["evidence"]
        }
    )
    return row


def _curate_selected(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_index = _source_index(candidates)
    rows = {row["id"]: deepcopy(row) for row in candidates if row["id"] in SELECTED_IDS}

    def edit(
        candidate_id: str,
        *,
        question: str,
        category: str,
        difficulty: str,
        task_type: str | None = None,
        answerability: str | None = None,
        claims: list[dict[str, Any]] | None = None,
        pmids: tuple[str, ...] | None = None,
        hard_negative_pmids: tuple[str, ...] | None = None,
        hard_negative_chunks: list[str] | None = None,
        missing_evidence: list[str] | None = None,
        forbidden_claims: list[str] | None = None,
    ) -> None:
        row = rows[candidate_id]
        row["question"] = question
        row["category"] = category
        row["difficulty_reason"] = difficulty
        if task_type is not None:
            row["task_type"] = task_type
        if answerability is not None:
            row["answerability"] = answerability
        if claims is not None:
            row["gold_claims"] = claims
        if pmids is not None:
            row["sources"] = _sources(source_index, *pmids)
        if hard_negative_pmids is not None:
            row["sources"] = _sources(
                source_index, *hard_negative_pmids, role="hard_negative"
            )
        if hard_negative_chunks is not None:
            row["hard_negative_chunk_ids"] = hard_negative_chunks
        if missing_evidence is not None:
            row["missing_evidence"] = missing_evidence
        if forbidden_claims is not None:
            row["forbidden_claims"] = forbidden_claims

    # Ten single-evidence questions.  Each asks for one result contained in one
    # source span and keeps the original study context in the wording.
    edit(
        "VMG-001",
        question="In the cross-sectional pilot study of veterans, what cardiac MRI findings in people living with HIV suggested myocardial fibrosis and subclinical dysfunction?",
        category="cardio_infectious",
        difficulty="The answer must preserve that these were cross-sectional pilot findings described as suggestive rather than causal.",
        forbidden_claims=["HIV was proven to cause myocardial fibrosis."],
    )
    edit(
        "VMG-054",
        question="Among fetuses already diagnosed with growth restriction at the anatomy scan, was a gestational-age discrepancy greater than 10 days associated with small-for-gestational-age birth?",
        category="womens_health",
        difficulty="The answer must distinguish the higher observed SGA percentage from a statistically nonsignificant adjusted association with a wide confidence interval.",
        task_type="single_evidence",
        claims=[
            _claim(
                "C1",
                "The >10-day group had more SGA neonates (53.9% vs 45.5%), but the association was not statistically significant before or after adjustment (RR 1.40, 95% CI 0.416-4.708; aRR 1.48, 95% CI 0.428-5.091).",
                (
                    "pubmed:41871491:0",
                    "GA > 10 days had more SGA neonates (53.9% vs 45.5%) but these findings were not significant (RR 1.40, 95% CI 0.416, 4.708) and persisted in multivariable analysis, (aRR 1.48, 95% CI 0.428, 5.091).",
                    "direct",
                    None,
                ),
            )
        ],
        pmids=("41871491",),
        forbidden_claims=["A discrepancy greater than 10 days significantly increased SGA risk."],
    )
    edit(
        "VMG-017",
        question="In the multicenter retrospective sTBI study, how did risk-stratified resource allocation change ICU bed occupancy, and what happened to overall prognosis?",
        category="critical_care",
        difficulty="The answer must pair the utilization change with the reported absence of a negative effect on overall prognosis.",
        claims=[
            _claim(
                "C1",
                "Risk-stratified allocation reduced ICU bed occupancy from 89.2% to 75.6% and improved specialized nursing-resource use, while overall prognosis was not negatively affected.",
                (
                    "pubmed:42065801:0",
                    "The resource allocation strategy guided by risk stratification significantly reduced ICU bed occupancy rate from 89.2% to 75.6% and improved the utilization efficiency of specialized nursing resources, while the overall prognosis of patients was not negatively affected.",
                    "direct",
                    None,
                ),
            )
        ],
    )
    edit(
        "VMG-025",
        question="In the thyroid-ultrasound AI meta-analysis, did cytology versus histology as the reference standard significantly explain diagnostic-accuracy differences in mixed-effects meta-regression?",
        category="endocrinology",
        difficulty="The answer must distinguish a suggestive subgroup difference from the nonsignificant meta-regression result.",
        forbidden_claims=["Cytology was proven to be a superior reference standard."],
    )
    edit(
        "VMG-035",
        question="For surveillance of local nasopharyngeal carcinoma recurrence, what sensitivities were reported for nasopharyngeal-brush versus plasma EBV DNA testing?",
        category="oncology",
        difficulty="The source reports a paired diagnostic comparison; the exact sensitivity values and direction must be retained.",
        task_type="single_evidence",
        claims=[
            _claim(
                "C1",
                "Nasopharyngeal-brush EBV DNA testing had 95.7% sensitivity versus 61.4% for plasma EBV DNA testing.",
                (
                    "pubmed:41631491:0",
                    "Nasopharyngeal brush EBV DNA detection demonstrated significantly higher sensitivity (95.7%) for detecting recurrence compared to plasma EBV DNA testing (61.4%), with comparable specificity (94.0% vs.",
                    "direct",
                    None,
                ),
            )
        ],
    )
    edit(
        "VMG-041",
        question="In the sodium-iodate mouse model of age-related macular degeneration, how much did high-dose Zhujing Pill restore retinal thickness relative to the untreated model group?",
        category="ophthalmology",
        difficulty="The question is deliberately limited to the preclinical mouse-model outcome and must not be answered as human efficacy.",
        claims=[
            _claim(
                "C1",
                "High-dose Zhujing Pill restored retinal thickness by 84.76% versus the AMD model group (P < 0.001).",
                (
                    "pubmed:41967781:0",
                    "In vivo evaluations, compared with the AMD model group, high-dose ZJP treatment increased the time spent in the dark chamber by 90.51% (P < 0.001), significantly restored retinal thickness by 84.76% (P < 0.001), and improved the histopathological score by 52.88% (P < 0.001).",
                    "direct",
                    None,
                ),
            )
        ],
        forbidden_claims=["Zhujing Pill has demonstrated clinical efficacy in people with AMD."],
    )
    edit(
        "VMG-043",
        question="In the single-center CT-versus-TEE pathway study before cardioversion, what was the observed acute-kidney-injury difference, and was it statistically significant?",
        category="cardiovascular_imaging",
        difficulty="The answer must include the confidence interval and P value so a numerical difference is not mistaken for a significant effect.",
        task_type="single_evidence",
        claims=[
            _claim(
                "C1",
                "Acute kidney injury occurred in 8.1% of CT patients and 3.9% of TEE patients; the +4.2 percentage-point risk difference was not statistically significant (95% CI -1.4 to +12.9; P=0.21).",
                (
                    "pubmed:42041167:0",
                    "Acute kidney injury occurred in 8.1% of CT patients compared with 3.9% in the TEE group (absolute risk difference +4.2%, 95% CI: -1.4% to +12.9%; OR 2.18, 95% CI: 0.60-7.45; p = 0.21).",
                    "direct",
                    None,
                ),
            )
        ],
    )
    edit(
        "VMG-049",
        question="After thermal ablation for HCC, what 1-, 2-, and 3-year recurrence-free survival rates were observed in the model-predicted high- and low-MVI-risk groups?",
        category="hepatology",
        difficulty="The answer requires six time-specific percentages and must keep model-predicted risk separate from pathologically confirmed invasion.",
    )
    edit(
        "VMG-065",
        question="In the prospective low-kVp cerebral CTA study, how did DL-based contrast enhancement and denoising change arterial attenuation and image noise?",
        category="radiology",
        difficulty="The answer must identify both image-quality directions and avoid treating volunteer imaging metrics as clinical outcomes.",
        claims=[
            _claim(
                "C1",
                "Applying DL-ACE and DL-DN increased arterial attenuation by 45.4% and decreased image noise by 34.5%, with higher CNR and subjective image quality than the 100-kVp group.",
                (
                    "pubmed:42062228:0",
                    "After applying DL-ACE and DL-DN in Groups B and C, arterial attenuation increased by 45.4% and image noise decreased by 34.5%, resulting in significantly higher arterial attenuation, CNR, and subjective image quality compared with Group A (<i>P</i> < 0.001).",
                    "direct",
                    None,
                ),
            )
        ],
    )
    edit(
        "VMG-014",
        question="What discrimination, sensitivity, and specificity did the catheter-related-thrombosis nomogram report in the oncology cohort?",
        category="oncology",
        difficulty="The answer must report model-performance metrics without converting them into a treatment recommendation or a claim of prospective utility.",
        task_type="single_evidence",
        claims=[
            _claim(
                "C1",
                "The nomogram reported an AUC of 0.866 (95% CI 0.837-0.895), sensitivity of 70.33%, and specificity of 85.89%.",
                (
                    "pubmed:42077124:0",
                    "A nomogram integrating these variables demonstrated good discrimination (area under the curve = 0.866, 95% CI: 0.837-0.895), with a sensitivity of 70.33% and a specificity of 85.89%.",
                    "direct",
                    None,
                ),
            )
        ],
        pmids=("42077124",),
        forbidden_claims=["The nomogram was prospectively shown to reduce catheter-related thrombosis."],
    )

    # Fifteen within-document synthesis questions.
    edit(
        "VMG-002",
        question="In the single-arm DYNAMITE study, what structural changes were measured at 9 months and what major adverse cardiac events were observed by 24 months after DynamX implantation?",
        category="interventional_cardiology",
        difficulty="The answer combines an imaging endpoint and later clinical-event count while preserving the single-arm design.",
        claims=[
            _claim(
                "C1",
                "From postprocedure to 9 months, mean device area increased by 0.37 mm² (P=0.010), while mean in-device lumen area decreased by 0.70 mm² (P<0.001).",
                (
                    "pubmed:41765253:0",
                    "At 9-month follow-up, mean device area increased to 8.53 ± 1.71 mm<sup>2</sup> (absolute difference: 0.37 ± 0.99 mm²; relative difference: 5.52 ± 14.8%, p = 0.010), and mean in-device lumen area decreased to 7.57 ± 1.86 mm<sup>2</sup> (absolute difference: -0.70 ± 1.09 mm²; relative difference: -7.98 ± 14.9%, p <0.001).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "By 24 months, 4 patients (7.4%) had major adverse cardiac events, including one target-vessel myocardial infarction and four target-lesion revascularization events.",
                (
                    "pubmed:41765253:0",
                    "At 24-month follow-up, 4 patients (7.4%) experienced major adverse cardiac events, including one target vessel myocardial infarction and 4 target lesion revascularization events.",
                    "direct",
                    None,
                ),
            ),
        ],
        forbidden_claims=["DynamX was superior to a drug-eluting stent comparator."],
    )
    edit(
        "VMG-003",
        question="In the Behçet's disease echocardiography study, which GLS, RV free-wall strain, and coronary-sinus-flow findings differed from controls, and what performance did the combined GLS-CSF model achieve?",
        category="cardiovascular_imaging",
        difficulty="The answer must integrate group differences with diagnostic-model performance from the same cross-sectional study.",
        claims=[
            _claim(
                "C1",
                "Absolute GLS, absolute RV free-wall strain, and the CSF index were all significantly lower in the Behçet's disease group than in controls.",
                (
                    "pubmed:42054004:0",
                    "Compared to the control group, GLS (absolute; 18.51 ± 2.02 vs. 19.53 ± 1.71, p = 0.018), right ventricular free wall strain (RVFWS; absolute) (25.35 ± 3.32 vs. 26.87 ± 3.01, p = 0.047), and CSF index (2.62 ± 0.99 vs. 3.58 ± 1.17 mL/min/g, p < 0.001) were found to be significantly lower in the BD group.",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "The combined GLS-CSF model had AUC 0.759, sensitivity 80.9%, and specificity 65.7%, outperforming either measure used alone.",
                (
                    "pubmed:42054004:0",
                    "Furthermore, the combined model using GLS and the CSF index exhibited a higher diagnostic performance in distinguishing BD patients compared to their individual use (AUC: 0.759, 80.9% sensitivity, 65.7% specificity).",
                    "direct",
                    None,
                ),
            ),
        ],
    )
    edit(
        "VMG-010",
        question="In the thermal-ablation HCC cohort, how did model-predicted MVI risk stratify recurrence-free survival, and which variables remained independently associated with RFS?",
        category="hepatology",
        difficulty="The response must combine time-specific survival estimates with adjusted hazard ratios and call the model result prognostic rather than treatment predictive.",
    )
    edit(
        "VMG-011",
        question="How was SCAR-Net built and validated for distinguishing postoperative breast scars from recurrent lesions, and how much did it change radiologist performance?",
        category="oncology",
        difficulty="The answer must link multicenter development scale and model components to the reported assisted-reader metrics.",
        claims=[
            _claim(
                "C1",
                "SCAR-Net was developed from 34,376 ultrasound images from 5,710 patients at four hospitals and used scar-recurrence feature-enhancer and boundary-sensitive attention modules.",
                (
                    "pubmed:42023157:0",
                    "Using 34,376 ultrasound images from 5,710 patients across four hospitals, we developed a model incorporating scar-recurrence feature enhancer and boundary-sensitive attention network modules.",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "With SCAR-Net assistance, radiologist AUC rose from 0.810-0.824 to 0.939-0.942, sensitivity from 0.775-0.782 to 0.934-0.941, and specificity from 0.839-0.872 to 0.935-0.950.",
                (
                    "pubmed:42023157:0",
                    "In multicenter validation, SCAR-Net significantly improved radiologists' diagnostic performance, increasing AUC from 0.810-0.824 to 0.939-0.942, sensitivity from 0.775-0.782 to 0.934-0.941, and specificity from 0.839-0.872 to 0.935-0.950 (all <i>p</i> < 0.001).",
                    "direct",
                    None,
                ),
            ),
        ],
    )
    edit(
        "VMG-018",
        question="In adenomyosis with dysmenorrhea, which regional-homogeneity changes distinguished patients from controls, and which pain or emotional measures correlated with the altered regions?",
        category="womens_health",
        difficulty="The answer combines case-control imaging differences with region-specific symptom correlations without making a causal claim.",
        claims=[
            _claim(
                "C1",
                "Patients had increased static and dynamic ReHo in the right fusiform gyrus and decreased values in both supramarginal gyri.",
                (
                    "pubmed:41614571:0",
                    "Compared with HCs, AMD patients showed significantly increased sReHo and dReHo in the right fusiform gyrus (FFG; <i>d</i> = 0.84, <i>p</i> < 0.001) and decreased values in the bilateral supramarginal gyri (SMG; <i>d</i> = -0.73, <i>p</i> < 0.001).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "Right fusiform ReHo correlated with pain severity and anxiety, while right hippocampal and cerebellar abnormalities correlated with anxiety and depression scores.",
                (
                    "pubmed:41614571:0",
                    "Importantly, ReHo in the right FFG positively correlated with pain severity (visual analogue scale (VAS): <i>r</i> = 0.539-0.797, <i>p</i> < 0.001) and anxiety (Hamilton Anxiety Scale (HAMA): <i>r</i> = 0.442, <i>p</i> = 0.001).",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41614571:0",
                    "Abnormalities in the right hippocampus and cerebellum were also significantly associated with anxiety and depression scores (<i>r</i> = 0.555-0.861, all <i>p</i> < 0.001).",
                    "direct",
                    None,
                ),
            ),
        ],
    )
    edit(
        "VMG-019",
        question="Among adults with epilepsy in Southern Kazakhstan, which seizure-history and MRI features were associated with drug-resistant rather than drug-sensitive epilepsy?",
        category="neurology",
        difficulty="The answer must retain observational association language and the MRI effect estimate.",
        forbidden_claims=["The associated factors were proven causes of drug resistance."],
    )
    edit(
        "VMG-026",
        question="After STEMI, how was left-atrial reservoir strain below 23% associated with heart-failure events across LVEF strata, and what risk-stratification conclusion did the study draw?",
        category="heart_failure",
        difficulty="The answer must integrate the threshold-specific effect with the study's qualified clinical interpretation.",
    )
    edit(
        "VMG-027",
        question="How did cerebellar glutamate-sensitive MRI measurements and cerebellar volume differ across resting-tremor PD, akinetic-rigid PD, and healthy controls?",
        category="neurology",
        difficulty="The answer must keep metabolic and volumetric findings separate and must not invent a glutamate result for the akinetic-rigid group.",
        claims=[
            _claim(
                "C1",
                "Resting-tremor-predominant PD showed elevated MTRasym in the dentate nucleus and cerebellar hemisphere, consistent with increased glutamate concentrations.",
                (
                    "pubmed:41687710:0",
                    "Results demonstrated significantly elevated MTR<sub>asym</sub> values in the dentate nucleus and cerebellar hemisphere of PDRT patients (*p*<0.05), indicative of increased glutamate concentrations.",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "The resting-tremor group had reduced cerebellar volume versus healthy controls, whereas the akinetic-rigid group had no significant volumetric difference.",
                (
                    "pubmed:41687710:0",
                    "Concurrently, PDRT exhibited reduced cerebellar volumes compared to HCs, whereas PDAR showed no significant volumetric differences.",
                    "direct",
                    None,
                ),
            ),
        ],
    )
    edit(
        "VMG-042",
        question="In postmenopausal women, which extracellular-vesicle and metabolite patterns accompanied coronary artery calcification, and how did prior preeclampsia relate to later CVD or CKD within the CAC-positive group?",
        category="womens_health",
        difficulty="The answer must synthesize biomarker differences with a subgroup risk association from a small observational sample.",
        forbidden_claims=["The biomarkers or preeclampsia were proven to cause later CVD or CKD."],
    )
    edit(
        "VMG-051",
        question="In the retrospective vacuum-assisted-birth cohort, which factors were independently associated with assisted vaginal birth, and how did mid/low-cavity applications differ from perineal applications?",
        category="womens_health",
        difficulty="The answer requires both the adjusted risk-factor pattern and the application-level outcome comparison.",
        claims=[
            _claim(
                "C1",
                "Older maternal age, mode of conception, epidural analgesia, later gestational age, and male newborn sex were positively associated with assisted vaginal birth, whereas parity was inversely associated.",
                (
                    "pubmed:42036403:0",
                    "After adjusting for all factors considered, maternal age (OR 1.04, 95% CI: 1.01-1.07), mode of conception (OR 1.58, 95% CI: 1.07-2.33), epidural analgesia (OR 6.25, 95% CI: 3.05-12.80), gestational age (OR 1.48, 95% CI: 1.31-1.67), and newborn male sex (OR 1.35, 95% CI: 1.06-1.73) were positively associated with AVB, whereas parity (OR 0.20, 95% CI: 0.14-0.29) was inversely associated.",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "Compared with perineal applications, mid/low-cavity applications had greater blood loss, a higher episiotomy rate (95.5% vs 88.2%), and more ultrasound use (62.4% vs 23.5%).",
                (
                    "pubmed:42036403:0",
                    "Compared to perineal applications, mid/low pelvic applications were associated with greater blood loss (<i>p</i> < 0.01), higher episiotomy rate (95.5% vs 88.2%, <i>p</i> = 0.03), and increased ultrasound use (62.4% vs 23.5%, <i>p</i> < 0.01).",
                    "direct",
                    None,
                ),
            ),
        ],
    )
    edit(
        "VMG-058",
        question="For the CT-based NSCLC survival model, how did performance compare with benchmark 3D networks on LUNG1, and what did transfer learning add on the smaller private dataset?",
        category="pulmonology",
        difficulty="The answer must keep the public-dataset comparison separate from the private-dataset transfer-learning increment.",
        claims=[
            _claim(
                "C1",
                "On LUNG1, the method outperformed the compared 3D networks and variants with a mean time-dependent C-index of 0.584 over tenfold cross-validation.",
                (
                    "pubmed:41492270:0",
                    "Our approach was compared to benchmark 3D networks and two variants of our methodology: on the LUNG1 it outperformed the competitors achieving a mean <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><msup><mi>C</mi> <mrow><mi>td</mi></mrow> </msup> </math> -index of 0.584 over tenfold cross-validation.",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "On the private dataset, transfer learning increased the time-dependent C-index by 0.076 relative to the model without transfer learning.",
                (
                    "pubmed:41492270:0",
                    "Finally, we used transfer learning on the private dataset, showing that it can significantly enhance performance in limited data scenarios, increasing the <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><msup><mi>C</mi> <mrow><mi>td</mi></mrow> </msup> </math> -index by 0.076 compared to model without transfer learning.",
                    "direct",
                    None,
                ),
            ),
        ],
    )
    edit(
        "VMG-059",
        question="In the prospective thoracic ASRT study, what did online adaptation change in target coverage and organ-at-risk dose, and what treatment-time and toxicity results were reported?",
        category="pulmonology",
        difficulty="The answer combines dosimetry, workflow duration, and observed toxicity without converting a single-arm study into a comparative clinical-effectiveness claim.",
        claims=[
            _claim(
                "C1",
                "Online adaptation improved median V100% target coverage relative to predicted plans (+0.1%±2.5% vs -2.6%±7.3%; P<0.001) without a significant OAR-dose difference.",
                (
                    "pubmed:42058324:0",
                    "Online adaptation significantly improved target coverage (median V100% change: +0.1%±2.5%) compared to predicted plans (-2.6%±7.3%, p < 0.001), without significant differences in OAR doses.",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "Median total treatment time was 28 minutes, the ASRT workflow averaged 5 minutes, no grade 3 or higher toxicities occurred, and grade 2 events occurred in 10.3%.",
                (
                    "pubmed:42058324:0",
                    "Median total treatment time was 28 min, with an average 5-minute ASRT workflow. No grade ≥ 3 toxicities were reported; grade 2 events occurred in 10.3% of patients.",
                    "direct",
                    None,
                ),
            ),
        ],
        forbidden_claims=["ASRT improved survival compared with nonadaptive radiotherapy."],
    )
    edit(
        "VMG-067",
        question="During and after pregnancy in women with LV outflow-tract obstruction, which severity measurements predicted postpartum intervention, and how did inherently severe versus transiently severe obstruction differ?",
        category="womens_health",
        difficulty="The answer must combine time-specific hazard ratios with the distinction between baseline severity and transient gestational elevation.",
        claims=[
            _claim(
                "C1",
                "Baseline, antenatal, and postpartum LVOTO severity predicted postpartum intervention, with hazard ratios 29.6, 21.1, and 18.3, respectively (all P<0.01).",
                (
                    "pubmed:41698552:0",
                    "Predictors of the primary endpoint were baseline LVOTO severity (HR = 29.6, p < 0.01), antenatal LVOTO severity (HR = 21.1, p < 0.01), and postpartum LVOTO severity (HR = 18.3, p < 0.01).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "Inherently severe baseline LVOTO was associated with higher postpartum-intervention risk, whereas transiently severe LVOTO during pregnancy was not.",
                (
                    "pubmed:41698552:0",
                    "While patients with inherently severe LVOTO at baseline (n = 8, 10%) had a significantly increased risk of postpartum intervention, those with transiently severe LVOTO during pregnancy did not.",
                    "direct",
                    None,
                ),
            ),
        ],
    )
    edit(
        "VMG-074",
        question="After reverse total shoulder arthroplasty, how did high versus low preoperative fatty infiltration relate to final forward-elevation and external-rotation strength, and to improvement from baseline?",
        category="musculoskeletal",
        difficulty="The answer must distinguish lower final strength from the absence of a significant between-group difference in strength improvement.",
        claims=[
            _claim(
                "C1",
                "At final follow-up, the high-fatty-infiltration group had lower forward-elevation and external-rotation strength than the low-infiltration group.",
                (
                    "pubmed:41248236:0",
                    "At the final follow-up, patients in the high FI group demonstrated significantly reduced strength in both forward elevation ( P = 0.049) and ER ( P = 0.007) compared with those in the low FI group.",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "The groups did not differ significantly in preoperative-to-postoperative improvement in forward-elevation or external-rotation strength.",
                (
                    "pubmed:41248236:0",
                    "However, the mean improvement in muscle strength from preoperative to postoperative evaluation in forward elevation and ER showed no significant difference between two groups ( P = 0.559, 0.675, respectively).",
                    "direct",
                    None,
                ),
            ),
        ],
    )
    edit(
        "VMG-075",
        question="For acute hand and wrist trauma, how did adding calcium-suppression images to conventional CT affect reader sensitivity, specificity, and additional fracture detection versus CT alone and X-ray?",
        category="musculoskeletal",
        difficulty="The answer must synthesize reader-level sensitivity, near-perfect specificity, and incremental detections across three image strategies.",
        claims=[
            _claim(
                "C1",
                "CT plus calcium suppression increased sensitivity to 94.3% and 97.2%, versus 88.2% and 90.1% for CT alone and 72.6% and 75.9% for X-ray, for the two readers.",
                (
                    "pubmed:41653331:0",
                    "Sensitivity was increased significantly for both readers in CT + CaSupp (94.3 and 97.2%) compared to conventional CT (88.2 and 90.1%, p < .01) and X-ray (72.6 and 75.9%, p < .01).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "Specificity remained 99.9%-100% with CT plus calcium suppression, and the two readers detected 13 and 15 additional fractures versus CT alone.",
                (
                    "pubmed:41653331:0",
                    "Specificity was 99.9 and 100% for CT + CaSupp, 99.8 and 99.9% for conventional CT, and 99.4 and 99.9% for X-ray. Reader 1 detected 13 additional fractures in CT + CaSupp compared to conventional CT alone, while Reader 2 detected 15 additional fractures.",
                    "direct",
                    None,
                ),
            ),
        ],
    )

    # Ten cross-document comparisons.  Each compares like concepts while
    # preserving study-specific populations, designs, and endpoints.
    edit(
        "VMG-004",
        question="Across the chronic-heart-failure and post-STEMI studies, how was reduced atrial reservoir strain associated with later adverse or heart-failure-related events?",
        category="heart_failure",
        difficulty="The answer must keep the two populations and strain definitions distinct rather than treating the estimates as head-to-head.",
        forbidden_claims=["The two strain strategies were directly compared in one trial."],
    )
    edit(
        "VMG-012",
        question="What target-specific tumor-localization evidence was reported for the LGR5 tracer [68Ga]Ga-LTP-02 and for the Nectin4 probes in their respective preclinical models?",
        category="oncology",
        difficulty="The answer compares target localization across two preclinical programs without ranking tracers tested in different models.",
        forbidden_claims=[
            "Either tracer has proven diagnostic accuracy in patients.",
            "The two tracers were evaluated head-to-head.",
        ],
    )
    edit(
        "VMG-013",
        question="How did AI assistance change diagnostic performance in the simulated prostate-MRI triage workflow and in multicenter breast-ultrasound scar-versus-recurrence assessment?",
        category="clinical_ai",
        difficulty="The studies report different imaging tasks and baselines; the answer must describe each within-study comparison without declaring one model superior.",
        claims=[
            _claim(
                "C1",
                "In simulated prostate-MRI triage, the AI pathway maintained sensitivity at 89.0% while increasing specificity by 11.5 percentage points to 69.2% versus the conventional pathway.",
                (
                    "pubmed:41960994:0",
                    "The AI-driven pathway maintained comparable sensitivity (89.0%; 95% CI: 85.0, 93.0; <i>P</i> = .36) but improved specificity by 11.5%, reaching 69.2% (95% CI: 64.4, 74.0; <i>P</i> < .001).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "In multicenter breast ultrasound, SCAR-Net assistance increased radiologist AUC, sensitivity, and specificity to 0.939-0.942, 0.934-0.941, and 0.935-0.950, respectively.",
                (
                    "pubmed:42023157:0",
                    "In multicenter validation, SCAR-Net significantly improved radiologists' diagnostic performance, increasing AUC from 0.810-0.824 to 0.939-0.942, sensitivity from 0.775-0.782 to 0.934-0.941, and specificity from 0.839-0.872 to 0.935-0.950 (all <i>p</i> < 0.001).",
                    "direct",
                    None,
                ),
            ),
        ],
        pmids=("41960994", "42023157"),
        forbidden_claims=["One AI system outperformed the other across studies."],
    )
    edit(
        "VMG-021",
        question="How did the fMRI acceleration method and the MR anisotropic-diffusion filters address different image-reconstruction problems, and what validation result did each report?",
        category="radiology",
        difficulty="The answer must distinguish acquisition acceleration from post-acquisition Rician-noise filtering and avoid a cross-study performance ranking.",
        claims=[
            _claim(
                "C1",
                "The fMRI method combined through-plane and in-plane acceleration with image shifts and 2D Hadamard encoding; in simulated and experimental data it reduced scan time while increasing SNR and CNR in regions of interest.",
                (
                    "pubmed:41951145:0",
                    "Multiple image-shift strategies and a 2D Hadamard encoding scheme are used to increase encoding diversity and reduce slice leakage.",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41951145:0",
                    "By applying our approach to both simulated and experimental fMRI data, we successfully reduced total scan time while achieving a higher signal-to-noise ratio (SNR) and contrast-to-noise ratio (CNR) in regions of interest (ROI).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "The MR filters used adaptively optimized anisotropic-diffusion models and a Rician-noise estimator; tests on synthetic and real MR images reported noise reduction with preservation of important details.",
                (
                    "pubmed:41903663:0",
                    "These filtering models incorporate a robust Rician noise estimator and an unbiased variant of filters based on anisotropic diffusion, whose parameters are all calculated simultaneously and optimized adaptively through the introduction of the PSO algorithm and the use of reference-based and no-reference quality metrics (PSNR, SSIM and Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)).",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41903663:0",
                    "The various tests performed on synthetic and real MR images provided excellent results compared to some other published methods, thus demonstrating the high performance of the proposed filters in terms of noise reduction and preservation of important details.",
                    "direct",
                    None,
                ),
            ),
        ],
        forbidden_claims=["One method was more accurate than the other."],
    )
    edit(
        "VMG-029",
        question="What did the two retrospective studies report about microwave ablation for benign thyroid nodules, and how do their populations and follow-up constrain comparison of outcomes?",
        category="endocrinology",
        difficulty="One study covers 123 nodules larger than 10 mL with longer follow-up; the other compares 12 cooled and 12 uncooled cases through six months.",
        claims=[
            _claim(
                "C1",
                "In 123 solid nodules larger than 10 mL, final median volume reduction was 80.73% after mean follow-up of 28.32 months, and no permanent sequelae were reported despite a 4.88% major-complication rate.",
                (
                    "pubmed:42003534:0",
                    "The median initial volume was 14.59 mL, final VRR reached 80.73% (mean follow-up 28.32 ± 7.91 months), with 91.87% achieving >50% VRR at 12th month.",
                    "direct",
                    None,
                ),
                (
                    "pubmed:42003534:0",
                    "The major complication rate was 4.88%, with no cases of permanent sequelae.",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "In the 24-patient cooled-versus-uncooled comparison, volume-reduction ratios were similar at three and six months, while complications occurred only in the cooled group; the authors cautioned that the sample and power settings limited interpretation.",
                (
                    "pubmed:41860416:0",
                    "Nodule volumes significantly decreased at the third and sixth month follow-ups, with similar volume reduction ratios between cMWA and uMWA (60.30 ± 8 vs 57.29 ± 8.2 at 3 mo and 78.03 ± 6 vs 77.55 ± 10 at 6 mo).",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41860416:0",
                    "One major and 2 minor complications (nodule rupture and superficial hematoma) occurred only in the cMWA group.",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41860416:0",
                    "However, observed differences in complications require cautious interpretation because of the limited sample size and the inherent difference in power settings.",
                    "direct",
                    None,
                ),
            ),
        ],
        forbidden_claims=["Cooled and uncooled systems were compared in the 123-patient study."],
    )
    edit(
        "VMG-036",
        question="What safety evidence did the patient-level femoral-vein-ultrasound study and the German AF-ablation trend study provide, and why are their estimates not directly comparable?",
        category="electrophysiology",
        difficulty="The answer must distinguish a contemporary cohort comparison of access complications from an administrative time trend in tamponade.",
        forbidden_claims=["Ultrasound localization caused the national decline in pericardial tamponade."],
    )
    edit(
        "VMG-044",
        question="How did a CT-based ccRCC T-stage prediction study differ from the tubulocystic RCC imaging series in clinical aim, sample size, and reported imaging value?",
        category="nephrology",
        difficulty="The answer compares a 218-patient prediction model with a seven-patient rare-tumor imaging description rather than pooling their results.",
        claims=[
            _claim(
                "C1",
                "The ccRCC study used 218 patients to build a preoperative T-stage model from perinephric fat stranding, RENAL score, and platelet count; its AUC was 0.867 versus 0.680 for radiologists.",
                (
                    "pubmed:41657050:0",
                    "A derivation cohort of 218 ccRCC patients with known pathological results and preoperative biomarker data was used to develop and validate a predictive model for preoperative T-stage.",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41657050:0",
                    "The performance of the predictive model built on these variables notably surpasses that of radiologists (AUC: 0.867 vs 0.680; delong test, <i>p</i> < 0.001).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "The tubulocystic RCC study described seven pathologically confirmed cases and found that ultrasound often showed a hyperechoic lesion with posterior acoustic enhancement, potentially adding information when CT or MRI was equivocal.",
                (
                    "pubmed:42044518:0",
                    "Seven patients (6 male, 1 female; mean age 61 ± 8 y) were included, with presentations of abdominal pain (n=2), hematuria (n=1), and incidental detection (n=4).",
                    "direct",
                    None,
                ),
                (
                    "pubmed:42044518:0",
                    "Although CT and MRI features may mimic cysts, US most commonly demonstrates a hyperechoic lesion with posterior acoustic enhancement and may provide important additional diagnostic information when other imaging findings are equivocal.",
                    "direct",
                    None,
                ),
            ),
        ],
    )
    edit(
        "VMG-052",
        question="In the two preclinical GOx nanoplatforms, how was glucose oxidase combined with the carrier and other mechanisms to intensify tumor treatment?",
        category="oncology",
        difficulty="The answer must keep manganese/CDT/autophagy and iron-oxide/photothermal mechanisms assigned to the correct platform and remain preclinical.",
        forbidden_claims=["Either nanoplatform has demonstrated safety or efficacy in patients."],
    )
    edit(
        "VMG-060",
        question="Across the two HFpEF studies, what did resting alveolar-membrane diffusing capacity indicate about exercise physiology, and what did systolic aortic regurgitation vena-contracta width indicate about prognosis?",
        category="cardiopulmonary",
        difficulty="The answer must distinguish physiologic correlations from an adjusted prognostic association in a separate cohort.",
        forbidden_claims=["The pulmonary measurement and regurgitation measure were compared in the same patients."],
    )
    edit(
        "VMG-077",
        question="How did post-processing improve image quality in hand/wrist spectral CT with calcium suppression and in low-kVp cerebral CTA with DL enhancement and denoising?",
        category="radiology",
        difficulty="The answer compares modality-specific within-study gains and must not equate fracture sensitivity with CTA attenuation or CNR.",
        claims=[
            _claim(
                "C1",
                "Adding calcium-suppression images increased fracture-detection sensitivity for both readers versus conventional CT alone and X-ray.",
                (
                    "pubmed:41653331:0",
                    "Sensitivity was increased significantly for both readers in CT + CaSupp (94.3 and 97.2%) compared to conventional CT (88.2 and 90.1%, p < .01) and X-ray (72.6 and 75.9%, p < .01).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "In low-kVp cerebral CTA, DL-ACE and DL-DN increased arterial attenuation by 45.4% and decreased image noise by 34.5%, improving CNR and subjective quality versus the 100-kVp group.",
                (
                    "pubmed:42062228:0",
                    "After applying DL-ACE and DL-DN in Groups B and C, arterial attenuation increased by 45.4% and image noise decreased by 34.5%, resulting in significantly higher arterial attenuation, CNR, and subjective image quality compared with Group A (<i>P</i> < 0.001).",
                    "direct",
                    None,
                ),
            ),
        ],
        pmids=("41653331", "42062228"),
        forbidden_claims=["The two post-processing methods were directly compared."],
    )

    # Five fully answerable limitation questions and five partial-evidence
    # questions.  Derived limitations explicitly state the inference rule.
    edit(
        "VMG-006",
        question="What did the pregnancy LVOTO cohort show about transient gradients and later intervention, and what does its observational design prevent concluding about treatment effects?",
        category="womens_health",
        difficulty="The answer must combine the observed natural history with a design-based causal boundary.",
        claims=[
            _claim(
                "C1",
                "For most pregnancies, gestational LVOT-gradient increases regressed to baseline; 11% required postpartum intervention over a median 6.3 years, with risk concentrated in severe preconception LVOTO rather than transient gestational severity.",
                (
                    "pubmed:41698552:0",
                    "Gestational increases in LVOT gradients were transient, regressing to baseline levels postpartum for most. Postpartum intervention was required in 11% of pregnancies within a median of 6.3 (4.4-9.2) years.",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41698552:0",
                    "While patients with inherently severe LVOTO at baseline (n = 8, 10%) had a significantly increased risk of postpartum intervention, those with transiently severe LVOTO during pregnancy did not.",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "Because this was a retrospective observational cohort, it cannot show that treating or not treating a gestational gradient changes postpartum outcomes.",
                (
                    "pubmed:41698552:0",
                    "A retrospective, observational study was conducted between 2009 and 2023 in two tertiary centres.",
                    "derived",
                    "A nonrandomized retrospective exposure-outcome study can estimate associations and natural history but cannot identify the causal effect of an intervention it did not assign.",
                ),
            ),
        ],
        forbidden_claims=["Transient LVOTO should never be treated during pregnancy."],
    )
    edit(
        "VMG-038",
        question="What population-level changes followed the 2020 CMV trial in the French retrospective cohort, and why do those trends not by themselves prove that expanded screening is safe or beneficial?",
        category="infectious_disease",
        difficulty="The answer must report temporal trends while separating them from the causal effect established by a different randomized trial.",
        claims=[
            _claim(
                "C1",
                "From 2021-2023, systematic screening rose from 22.0% to 40.0%, treatment was used more often, and termination of pregnancy fell from 25.9% to 13.0% versus 2017-2020.",
                (
                    "pubmed:42076947:0",
                    "Compared with the period from 2017 to 2020, in the period from 2021 to 2023 there was a significant increase in both systematic CMV screening (from 22.0% to 40.0%; P = 0.001) and maternal requests for testing (from 0% to 4.2%; P = 0.02).",
                    "direct",
                    None,
                ),
                (
                    "pubmed:42076947:0",
                    "Among cases of maternal infection during the periconceptional period or in the first trimester, antiviral therapy (generally valacyclovir) was administered more frequently in the period from 2021 to 2023 (27.7% vs 59.8%; P < 0.0001).",
                    "direct",
                    None,
                ),
                (
                    "pubmed:42076947:0",
                    "The overall rate of termination of pregnancy (TOP) for early maternal CMV-PI was 20.7% (40/193 with known pregnancy outcome), with significantly fewer TOPs being performed in the period from 2021 to 2023 (25.9% vs 13.0%; P = 0.03).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "The retrospective before-versus-after cohort can show coincident changes but cannot isolate the causal effect of screening or treatment on pregnancy and neonatal outcomes.",
                (
                    "pubmed:42076947:0",
                    "In this retrospective, multicenter study, we retrieved data from the electronic database of the French National Reference Center for Herpesviruses from cases of maternal CMV infection diagnosed during pregnancy between January 2017 and December 2023, with known neonatal infection status (infected or non-infected) at birth.",
                    "derived",
                    "Temporal practice changes in a retrospective cohort are vulnerable to concurrent changes and selection; they do not randomize screening or treatment.",
                ),
            ),
        ],
        forbidden_claims=["The cohort proves that national systematic screening prevents congenital CMV."],
    )
    edit(
        "VMG-046",
        question="What association did the hemodialysis study find between AV-access flow and LV mass, and why can it not establish that lowering access flow will improve cardiac outcomes?",
        category="nephrology",
        difficulty="The answer must interpret a cross-sectional surrogate-endpoint association without converting it into an intervention claim.",
        claims=[
            _claim(
                "C1",
                "Higher AV-access flow was associated with a 22.83-unit higher LV mass index (95% CI 0.37-45.29; P=0.046), whereas dialysis vintage was not significantly associated.",
                (
                    "pubmed:41644509:0",
                    "Higher FV was significantly associated with increased LVMI (mean difference = 22.83; 95% confidence interval [CI]: 0.37 to 45.29; <i>p</i> = 0.046), whereas dialysis vintage was not (mean difference = 0.51; 95% CI: -0.41 to 1.45; <i>p</i> = 0.27).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "Its cross-sectional design establishes neither temporal causality nor whether an intervention to reduce flow would lower LV mass or clinical cardiovascular events.",
                (
                    "pubmed:41644509:0",
                    "In this multicenter cross-sectional study, we enrolled hemodialysis patients who had undergone both transthoracic echocardiography and measurement of AV access flow, between April 2014 and January 2024.",
                    "derived",
                    "Exposure and surrogate outcome were measured in a cross-sectional study with no assigned flow-reduction intervention or clinical-event comparison.",
                ),
            ),
        ],
        forbidden_claims=["Reducing AV-access flow was shown to prevent cardiovascular events."],
    )
    edit(
        "VMG-062",
        question="Why do the CABG data on TEE plus pulmonary-artery catheter use require comparator-specific interpretation, and what evidence is still needed before choosing a monitoring strategy?",
        category="cardiac_surgery",
        difficulty="The exposure has opposite-looking associations under different comparators in a retrospective administrative cohort.",
        claims=[
            _claim(
                "C1",
                "TEE plus PAC was associated with more 30-day mortality or major complications than neither modality (aRR 1.18), yet with lower 30-day mortality than TEE alone (aRR 0.59); PAC findings therefore depended on the comparator and outcome.",
                (
                    "pubmed:42069508:0",
                    "Adjusted analyses revealed a higher risk of the primary outcome for TEE + PAC (aRR 1.18; 95% CI, 1.11-1.25) and PAC alone (aRR 1.05; 95% CI, 1.02-1.09), while TEE alone was not significant (aRR 1.06; 95% CI, 0.99-1.14) versus neither modality.",
                    "direct",
                    None,
                ),
                (
                    "pubmed:42069508:0",
                    "When compared to TEE alone, TEE + PAC was associated with a lower risk of 30-day mortality (aRR 0.59; 95% CI, 0.36-0.96).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "The retrospective study does not establish which modality causes better outcomes; prospective studies and longer-term outcomes are still needed.",
                (
                    "pubmed:42069508:0",
                    "Prospective studies are needed to identify subgroups benefiting from specific modalities and evaluate longer-term outcomes.",
                    "direct",
                    None,
                ),
            ),
        ],
        forbidden_claims=["TEE plus PAC is proven safer than TEE alone for CABG."],
    )
    edit(
        "VMG-070",
        question="What did the small postmenopausal-women study associate with CAC and later CVD or CKD, and which sample and design limits constrain causal or general claims?",
        category="womens_health",
        difficulty="The answer must combine the subgroup association with the 29-versus-29 sample and avoid causal biomarker claims.",
        claims=[
            _claim(
                "C1",
                "Women with CAC had less favorable cardiovascular/metabolic profiles and distinct extracellular-vesicle and metabolite levels; within the CAC-positive group, prior preeclampsia was associated with fourfold higher later CVD or CKD risk.",
                (
                    "pubmed:41973512:0",
                    "Patients with, versus those without, CAC demonstrated <i>1</i>) less favorable cardiovascular and metabolic profiles; <i>2</i>) elevation in six EV populations, including those positive for tissue factor, CD3 (T-cells), SM22α (smooth muscle cells), Pref-1 (adipocytes), fatty acid-binding protein 4 (adipocytes/macrophages), and p16 (senescent cells); <i>3</i>) significantly higher levels of proline, allothreonine (amino acid metabolism), and ribitol (carbohydrate metabolism), and lower levels of lactic acid (carbohydrate metabolism); and <i>4</i>) significantly increased risk of developing CVD and chronic kidney disease (CKD) (<i>P</i> < 0.05 for all).",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41973512:0",
                    "In the CAC-positive group, women with PE versus those with normotensive pregnancy histories demonstrated a four times higher risk of developing cardiovascular events or CKD later in life (<i>P</i> = 0.028).",
                    "direct",
                    None,
                ),
            ),
            _claim(
                "C2",
                "The analysis included only 29 women with and 29 without CAC and observed rather than assigned biomarkers or preeclampsia history, limiting precision, generalizability, and causal interpretation.",
                (
                    "pubmed:41973512:0",
                    "Clinical data were obtained from medical records for postmenopausal women (median age 60 yr) free of cardiovascular events with (<i>n</i> = 29) and without (<i>n</i> = 29) CAC.",
                    "derived",
                    "A 58-person observational comparison, with the preeclampsia result confined to the CAC-positive subgroup, cannot establish causality and has limited precision and transportability.",
                ),
            ),
        ],
        forbidden_claims=["Preeclampsia or the measured biomarkers caused later CVD or CKD."],
    )
    edit(
        "VMG-007",
        question="What evidence did the small HIV pilot provide for myocardial fibrosis and right-ventricular dysfunction, and what longitudinal question remains unanswered?",
        category="cardio_infectious",
        difficulty="Part of the question is answerable from a 21-versus-20 cross-sectional pilot; disease trajectory is deliberately absent.",
        task_type="limitations_and_safety",
        claims=[
            _claim(
                "C1",
                "The cross-sectional pilot found higher GDF-15, greater extracellular volume, and lower right-heart function in 21 veterans with HIV versus 20 controls, suggesting increased fibrosis and subclinical dysfunction.",
                (
                    "pubmed:41791688:0",
                    "21 veterans with HIV (mean age 54 years; 71% White, 29% Black) and 20 controls (mean age 56 years; 70% White, 30% Black) were included.",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41791688:0",
                    "We found higher growth differentiation factor (GDF)-15 blood levels, increased extracellular volume, and lower right heart function on MRI among patients with HIV, suggesting the presence of increased myocardial fibrosis and subclinical myocardial dysfunction in this population.",
                    "direct",
                    None,
                ),
            )
        ],
        missing_evidence=[
            "The cross-sectional pilot does not show the trajectory of cardiac changes or whether HIV causes them; larger longitudinal evidence is missing."
        ],
        forbidden_claims=["The study proves that HIV causes progressive myocardial fibrosis."],
    )
    edit(
        "VMG-015",
        question="What sensitivity-specificity tradeoff did simulated prostate-MRI AI triage show, and what real-world clinical effect remains untested?",
        category="clinical_ai",
        difficulty="Diagnostic simulation results are present, but prospective workflow, downstream-biopsy, and patient-outcome effects are absent.",
        task_type="limitations_and_safety",
        claims=[
            _claim(
                "C1",
                "The simulated AI pathway maintained sensitivity at 89.0% versus 89.4% for radiologists (P=0.36) and increased specificity by 11.5 percentage points to 69.2% (P<0.001).",
                (
                    "pubmed:41960994:0",
                    "The AI-driven pathway maintained comparable sensitivity (89.0%; 95% CI: 85.0, 93.0; <i>P</i> = .36) but improved specificity by 11.5%, reaching 69.2% (95% CI: 64.4, 74.0; <i>P</i> < .001).",
                    "direct",
                    None,
                ),
            )
        ],
        missing_evidence=[
            "Because the workflow was simulated retrospectively, the passages do not establish prospective deployment effects on workload, biopsies, treatment, or patient outcomes."
        ],
        forbidden_claims=["The AI pathway was prospectively deployed and improved patient outcomes."],
    )
    edit(
        "VMG-031",
        question="How well did ultrasound AI models diagnose malignant thyroid nodules in the external-dataset meta-analysis, and which evidence needed for broad clinical adoption remains missing?",
        category="endocrinology",
        difficulty="Pooled diagnostic accuracy is available, while impact on clinical decisions and underrepresented settings is not.",
        task_type="limitations_and_safety",
        claims=[
            _claim(
                "C1",
                "Across 27 studies, pooled sensitivity was 87%, specificity 83%, and the summary AUC 91.9%; the authors still called for more multicenter, non-Asian, histology-based studies.",
                (
                    "pubmed:41477624:0",
                    "Twenty-seven studies comprising 146,332 patients and over 600,000 ultrasound images met inclusion criteria. Overall, pooled sensitivity was 87 % (95 % CI: 84-89 %) and specificity 83 % (95 % CI: 79-86 %). The summary operating point indicated a sensitivity of 88 % and specificity of 83 %, with an AUC of 91.9 % (95 % CI: 90.0-93.2 %).",
                    "direct",
                    None,
                ),
                (
                    "pubmed:41477624:0",
                    "These findings support the potential integration of AI into clinical thyroid nodule management, although further multicenter, non-Asian, and histology-based studies are warrantee.",
                    "direct",
                    None,
                ),
            )
        ],
        missing_evidence=[
            "The selected evidence does not show whether adopting these models changes biopsy, surgery, missed-cancer, or patient-outcome rates in routine care."
        ],
        forbidden_claims=["The meta-analysis proves that AI improves patient outcomes or should replace clinicians."],
    )
    edit(
        "VMG-055",
        question="What prognostic association did the HCC model show after thermal ablation, and what evidence is missing before using it to choose treatment?",
        category="hepatology",
        difficulty="The retrospective association is available, but no prospective model-guided treatment comparison is present.",
        task_type="limitations_and_safety",
        claims=[
            _claim(
                "C1",
                "Model-predicted high MVI risk was independently associated with recurrence-free survival after thermal ablation (HR 2.90, 95% CI 1.69-4.96), as was AFP-L3% (HR 1.34, 95% CI 1.06-1.69).",
                (
                    "pubmed:42047408:0",
                    "High MVI risk predicted by the model (HR = 2.90; 95% CI: 1.69-4.96) and AFP-L3% (HR = 1.34; 95% CI: 1.06-1.69) were independently associated with RFS in TA patients.",
                    "direct",
                    None,
                ),
            )
        ],
        missing_evidence=[
            "The retrospective single-center study does not test whether using the prediction to select ablation, resection, or another treatment improves outcomes; prospective impact validation is missing."
        ],
        forbidden_claims=["The model identifies which treatment will improve survival for an individual patient."],
    )
    edit(
        "VMG-079",
        question="What association linked 6MCTB performance with lumbar-multifidus activation in chronic low-back pain, and what treatment-effect evidence is absent?",
        category="musculoskeletal",
        difficulty="The evidence covers a physiologic association, not the effect of 6MCTB-guided rehabilitation on symptoms.",
        task_type="limitations_and_safety",
        claims=[
            _claim(
                "C1",
                "Poorer individual, direction-specific, and summed 6MCTB performance was consistently associated with reduced lumbar-multifidus activation in patients with chronic low-back pain.",
                (
                    "pubmed:41927242:0",
                    "Based on individual tests, direction-specific classification, and summation from the 6MCTB, poorer test performance was consistently associated with reduced LM activation in patients with CLBP.",
                    "direct",
                    None,
                ),
            )
        ],
        missing_evidence=[
            "The study does not compare 6MCTB-guided rehabilitation with usual care or report effects on pain, disability, or long-term function."
        ],
        forbidden_claims=["6MCTB-guided exercise was shown to improve pain or disability."],
    )

    # Five clinically plausible abstention cases.  Each hard negative is on the
    # same topic but lacks the requested treatment-effect or outcome comparison.
    edit(
        "VMG-008",
        question="Do the supplied atrial-strain studies show that using atrial-strain-guided management improves heart-failure outcomes compared with standard management?",
        category="heart_failure",
        difficulty="The passages contain prognostic associations but no randomized or comparative strain-guided management strategy.",
        claims=[],
        hard_negative_pmids=("41932653", "41894993"),
        hard_negative_chunks=["pubmed:41932653:0", "pubmed:41894993:0"],
        missing_evidence=[
            "No study in the supplied passages compares atrial-strain-guided management with standard management or estimates its effect on clinical outcomes."
        ],
        forbidden_claims=[
            "Atrial-strain-guided management reduces heart-failure events.",
            "A specific strain threshold should be used to choose therapy.",
        ],
    )
    edit(
        "VMG-016",
        question="Does the supplied prostate-MRI study establish that deploying AI triage reduces unnecessary biopsies or improves patient outcomes versus a radiologist-only workflow?",
        category="clinical_ai",
        difficulty="The hard-negative passage reports a simulated diagnostic workflow, not prospective downstream clinical effects.",
        claims=[],
        hard_negative_pmids=("41960994",),
        hard_negative_chunks=["pubmed:41960994:0"],
        missing_evidence=[
            "The simulation reports sensitivity, specificity, and examinations triaged but no prospective comparison of biopsy use, treatment, morbidity, or patient outcomes."
        ],
        forbidden_claims=["AI triage was shown to reduce biopsies or improve patient outcomes."],
    )
    edit(
        "VMG-040",
        question="Do the supplied French cohort data establish that expanding systematic maternal CMV screening reduces congenital CMV infection or childhood neurosensory impairment compared with no screening?",
        category="infectious_disease",
        difficulty="The passage describes changing practice and pregnancy termination, but not a causal population-level screening comparison for infant outcomes.",
        claims=[],
        hard_negative_pmids=("42076947",),
        hard_negative_chunks=["pubmed:42076947:0"],
        missing_evidence=[
            "No randomized or adequately controlled screening-policy comparison reports congenital-infection or childhood-neurosensory outcomes in the supplied evidence."
        ],
        forbidden_claims=["Systematic maternal CMV screening was proven to prevent congenital infection or disability."],
    )
    edit(
        "VMG-064",
        question="Does the supplied ASRT study show that adaptive skin radiotherapy improves local control or survival compared with nonadaptive thoracic radiotherapy?",
        category="pulmonology",
        difficulty="The passage reports outcomes from a prospective single-arm cohort but provides no concurrent nonadaptive comparator.",
        claims=[],
        hard_negative_pmids=("42058324",),
        hard_negative_chunks=["pubmed:42058324:0"],
        missing_evidence=[
            "A comparative nonadaptive-radiotherapy group or randomized trial is absent, so the effect of ASRT on local control or survival cannot be estimated."
        ],
        forbidden_claims=["ASRT improves local control or survival versus nonadaptive radiotherapy."],
    )
    edit(
        "VMG-080",
        question="Do the supplied 6MCTB data show that assigning rehabilitation according to movement-control-test results improves pain or disability compared with usual rehabilitation?",
        category="musculoskeletal",
        difficulty="The passage is an association study of test performance and muscle activation, not a treatment trial.",
        claims=[],
        hard_negative_pmids=("41927242",),
        hard_negative_chunks=["pubmed:41927242:0"],
        missing_evidence=[
            "No intervention comparison assigns 6MCTB-guided versus usual rehabilitation or reports comparative pain and disability outcomes."
        ],
        forbidden_claims=["6MCTB-guided rehabilitation improves pain or disability."],
    )

    return [_finalize_row(rows[candidate_id]) for candidate_id in SELECTED_IDS]


def _append_curator_events(
    review_log_path: Path,
    candidate_ids: list[str],
) -> None:
    events = load_review_events(review_log_path)
    for candidate_id in candidate_ids:
        status = current_status(events, candidate_id)
        if status.value == "adversarial_checked":
            selected = candidate_id in SELECTED_IDS
            event = ReviewEvent(
                question_id=candidate_id,
                from_status="adversarial_checked",
                to_status="adjudicated",
                actor="curator",
                created_at=datetime.now(timezone.utc),
                findings=(
                    []
                    if selected
                    else [REJECTION_REASONS[candidate_id]]
                ),
                decision="accept_after_manual_rewrite" if selected else "reject",
                resolution=(
                    "The curator reconciled the question, atomic claims, qualifiers, and exact quotes against the frozen source passage."
                    if selected
                    else REJECTION_REASONS[candidate_id]
                ),
            )
            append_review_event(review_log_path, event)
            events.append(event)
            status = event.to_status
        if candidate_id in SELECTED_IDS and status.value == "adjudicated":
            event = ReviewEvent(
                question_id=candidate_id,
                from_status="adjudicated",
                to_status="frozen",
                actor="curator",
                created_at=datetime.now(timezone.utc),
                findings=[],
                decision="freeze",
                resolution="Included in the manually curated 50-question v1 benchmark.",
            )
            append_review_event(review_log_path, event)
            events.append(event)


def _dataset_card(
    questions: list[BenchmarkQuestion],
    *,
    questions_sha256: str,
    snapshot_id: str,
) -> str:
    task_counts = Counter(row.task_type.value for row in questions)
    answer_counts = Counter(row.answerability.value for row in questions)
    domain_counts = Counter(row.category for row in questions)
    domain_text = ", ".join(
        f"{domain} ({count})" for domain, count in sorted(domain_counts.items())
    )
    return f"""# VeritasMed Evidence-Grounded Benchmark v1

This is a 50-question engineering benchmark for evidence-grounded medical-literature RAG. It is designed to test whether an agent retrieves relevant passages, reconstructs supported claims, preserves study limits, cites evidence, and abstains when the supplied literature cannot answer a question.

It is **not clinician-reviewed**, is not a medical exam, and must not be used to make patient-care decisions.

## What is frozen

- Corpus snapshot: `{snapshot_id}`
- Question file SHA-256: `{questions_sha256}`
- Split: 15 development questions and 35 holdout questions
- Tasks: {dict(task_counts)}
- Answerability: {dict(answer_counts)}
- Domains: {domain_text}

Candidate IDs are preserved in `curation_decisions.jsonl`; final IDs are contiguous and stable. The append-only `review_log.jsonl` preserves the Qwen reconstruction, MedGemma challenge, and curator decision for every candidate.

## How the questions were made

Evidence was selected before question wording. Eighty candidates were drafted from exact PubMed passages. `qwen3:8b` independently reconstructed answers and `medgemma1.5:4b` challenged population, intervention/exposure, comparator, outcome, time point, and uncertainty. Model agreement never accepted an item automatically.

The curator then read all 80 question-claim-evidence bundles, rejected 30, and manually rewrote the selected 50. Required claims use verbatim evidence spans. A `derived` span is permitted only when it includes the explicit reasoning rule, such as why a cross-sectional association cannot establish an intervention effect.

## Composition

- 10 single-evidence questions
- 15 within-document synthesis questions
- 10 cross-document comparison questions
- 10 limitations and safety questions, including five partially answerable cases
- 5 insufficient-evidence questions that require abstention

The five abstention cases use same-topic hard negatives. They ask for an intervention effect or clinical outcome that the supplied observational, diagnostic, or single-arm evidence does not provide.

## Scoring intent

Score retrieval, claim completeness, claim support, citation coverage, qualification preservation, and answerability separately. A high-quality answer may be a refusal when the evidence does not contain the requested comparison. Do not award credit for medically plausible facts that are absent from the frozen passages.

## Known limits

- The corpus is a local snapshot and is not a systematic review for any clinical question.
- Most questions use PubMed abstracts rather than full articles, so the gold answer is bounded to the frozen passage.
- The dataset was curated by an engineering reviewer with local-model assistance, without clinician adjudication.
- Some 2026 records describe small, retrospective, preclinical, or technical studies; questions explicitly preserve those limits.
- Public repository users can inspect holdout labels. The holdout split prevents development leakage inside this project, not determined leaderboard gaming.

## Files

- `questions.jsonl`: final 50 questions
- `curation_decisions.jsonl`: final-to-candidate mapping and rejection reasons
- `review_log.jsonl`: append-only machine and curator review history
- `manifest.json`: frozen input hashes and corpus identity policy
- `model_manifest.json`: local-review model provenance
"""


def curate(root: Path, output_dir: Path) -> dict[str, Any]:
    candidates_path = output_dir / "candidates.jsonl"
    review_log_path = output_dir / "review_log.jsonl"
    candidates = _read_jsonl(candidates_path)
    if len(candidates) != 80:
        raise SystemExit(f"expected 80 candidates, found {len(candidates)}")

    curated_dicts = _curate_selected(candidates)
    curated = [BenchmarkQuestion.model_validate(row) for row in curated_dicts]
    frozen, provenance = freeze_questions(
        curated,
        development_ids=DEVELOPMENT_IDS,
    )

    normalized_chunks = build_normalized_chunks(
        root / "data/raw/pubmed/abstracts.jsonl",
        root / "data/raw/pmc/full_texts.jsonl",
    )
    chunk_map = {row["chunk_id"]: row["text"] for row in normalized_chunks}
    errors = validate_questions(frozen, chunk_map)
    if errors:
        raise SystemExit("curated evidence validation failed:\n- " + "\n- ".join(errors))

    questions_path = output_dir / "questions.jsonl"
    _write_jsonl(
        questions_path,
        [row.model_dump(mode="json") for row in frozen],
    )
    questions_sha256 = sha256_file(questions_path)

    decisions: list[dict[str, Any]] = []
    inverse_provenance = {source: final for final, source in provenance.items()}
    for candidate in candidates:
        candidate_id = candidate["id"]
        selected = candidate_id in SELECTED_IDS
        decisions.append(
            {
                "candidate_id": candidate_id,
                "decision": "selected_after_manual_rewrite" if selected else "rejected",
                "final_id": inverse_provenance.get(candidate_id),
                "reason": (
                    "Manually reconciled against the frozen source; wording, atomic claims, qualifiers, and evidence boundaries are encoded in questions.jsonl."
                    if selected
                    else REJECTION_REASONS[candidate_id]
                ),
            }
        )
    _write_jsonl(output_dir / "curation_decisions.jsonl", decisions)

    _append_curator_events(review_log_path, [row["id"] for row in candidates])

    corpus_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    (output_dir / "dataset_card.md").write_text(
        _dataset_card(
            frozen,
            questions_sha256=questions_sha256,
            snapshot_id=corpus_manifest["snapshot_id"],
        ),
        encoding="utf-8",
        newline="\n",
    )
    result = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_snapshot_id": corpus_manifest["snapshot_id"],
        "questions_path": "data/benchmark/veritasmed_v1/questions.jsonl",
        "questions_sha256": questions_sha256,
        "question_count": len(frozen),
        "development_count": sum(row.split.value == "development" for row in frozen),
        "test_count": sum(row.split.value == "test" for row in frozen),
        "candidate_count": len(candidates),
        "selected_candidate_count": len(SELECTED_IDS),
        "selection_policy": "manual source-first curation; retrieval ranks not consulted",
        "clinician_reviewed": False,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    result = curate(root, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
