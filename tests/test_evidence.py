"""Source provenance and bounded repair behaviors, using fictional study data."""
import json
from unittest.mock import MagicMock, patch

from medrag.agent.evidence import (
    bind_claims, bind_components, missing_numeric_details, outline_status, repair_gaps, restore_numeric_quotes,
)
from medrag.retrieval.retriever import RetrievedChunk


def chunk(doc_id="1", text="Sensitivity was 92% vs. 84% in 30 participants; P < .001."):
    return RetrievedChunk(f"pubmed:{doc_id}:0", text, 0.9,
                          {"source": "pubmed", "doc_id": doc_id, "pmid": doc_id})


def outline():
    source = chunk()
    return bind_components([{
        "requirement": "Diagnostic performance and comparator",
        "status": "supported",
        "evidence": [{"chunk_id": source.chunk_id, "quote": source.text}],
        "required_details": ["92% vs. 84%", "30 participants", "P < .001"],
    }, {
        "requirement": "Prospective patient benefit", "status": "missing",
        "gap": "The retrieved study does not establish prospective patient benefit.",
    }], [], [source])


def test_complete_comparison_and_inequality_survive_binding():
    component = outline()[0]
    assert component["required_details"] == ["92% vs. 84%", "30 participants", "P < .001"]
    assert component["evidence"][0]["citation"] == "PMID:1"


def test_selected_sentence_ids_recover_exact_quotes_without_model_transcription():
    source = chunk()
    result = bind_components([{
        "requirement": "Performance", "status": "supported", "evidence_ids": ["E1"],
    }], [], [source])
    assert result[0]["evidence"][0]["quote"] == source.text


def test_comparative_sentence_keeps_its_preceding_reference_values():
    source = chunk(text="Readers achieved sensitivity of 84%. Assistance achieved comparable sensitivity of 85%.")
    result = bind_components([{
        "requirement": "Diagnostic change", "status": "supported", "evidence_ids": ["E2"],
    }], [], [source])
    assert "Readers achieved sensitivity of 84%." in result[0]["required_details"]
    assert len(result[0]["evidence"]) == 2


def test_a_preceding_numeric_method_detail_is_not_an_answer_requirement():
    source = chunk(text="The probe had purity of 99%. Uptake was higher in positive tumors than negative tumors.")
    result = bind_components([{
        "requirement": "Tumor localization", "status": "supported", "evidence_ids": ["E2"],
    }], [], [source])
    assert not result[0]["required_details"]
    assert len(result[0]["evidence"]) == 1


def test_explicit_cohort_counts_survive_an_omitted_model_population_field():
    source = chunk(text="We enrolled 31 patients and 29 controls. Sensitivity was 92% versus 84%.")
    result = bind_components([{
        "requirement": "Diagnostic comparison", "status": "supported", "evidence_ids": ["E2"],
    }], [], [source])
    assert "We enrolled 31 patients and 29 controls." in result[0]["required_details"]
    assert "Sensitivity was 92% versus 84%." in result[0]["required_details"]
    repaired = restore_numeric_quotes(result, [{
        "component_id": "C1", "text": "Sensitivity was 92% versus 84%", "cite": ["PMID:1"],
    }])
    assert "31 patients and 29 controls" in repaired[-1]["text"]


def test_html_isotopes_and_thousands_separators_do_not_trigger_duplicate_quotes():
    source = chunk(text="[<sup>64</sup>Cu]Cu-probe uptake was 4.2%. We enrolled 1,234 patients.")
    components = bind_components([{
        "requirement": "Uptake", "status": "supported", "evidence_ids": ["E1"],
        "required_details": ["[<sup>64</sup>Cu]Cu-probe uptake was 4.2%."],
    }], [], [source])
    claims = [{"component_id": "C1", "text": "[64Cu]Cu-probe uptake was 4.2% in 1234 patients.", "cite": ["PMID:1"]}]
    assert restore_numeric_quotes(components, claims) == claims
    claims[0]["text"] = "[64Cu]Cu-probe uptake was 4.2%."
    assert "1,234 patients" in restore_numeric_quotes(components, claims)[-1]["text"]


def test_model_population_field_cannot_add_unrelated_method_sentences():
    source = chunk(text="The probe uses isotope 64. Uptake was 4.2%.")
    result = bind_components([{
        "requirement": "Uptake", "status": "supported", "evidence_ids": ["E2"],
    }], [], [source], [{"chunk_id": source.chunk_id, "quote": "The probe uses isotope 64.",
                       "required_details": ["isotope 64"]}])
    assert len(result[0]["evidence"]) == 1
    assert not result[0]["required_details"]


def test_omitted_answer_does_not_mean_source_evidence_is_missing():
    from medrag.agent.nodes import check_faithfulness
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({
        "supported": True, "complete": False, "boundary_correct": True,
        "component_checks": [{"id": "C1", "passed": False, "evidence_status": "missing",
                              "unsupported_source_inference": False,
                              "gap": "Answer omitted performance", "correction": "Include the result"}],
    }))
    with patch("medrag.agent.nodes.make_llm_think", return_value=llm):
        result = check_faithfulness({"query": "Performance?", "answer": "No evidence",
                                    "answer_components": outline()[:1], "retrieved_chunks": [chunk()]})
    assert result["answer_components"][0]["status"] == "supported"
    assert result["repair_component_ids"] == ["C1"]


def test_last_check_removes_an_outcome_not_established_by_the_source():
    from medrag.agent.nodes import check_faithfulness
    component = dict(outline()[0], requirement="Fewer hospital admissions")
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({
        "supported": False, "complete": False, "boundary_correct": False,
        "component_checks": [{"id": "C1", "passed": False, "evidence_status": "missing",
                              "unsupported_source_inference": True,
                              "gap": "The answer incorrectly infers admissions from sensitivity.",
                              "correction": "Remove the admission claim"}],
    }))
    with patch("medrag.agent.nodes.make_llm_think", return_value=llm):
        result = check_faithfulness({
            "query": "Did it reduce admissions?", "answer": "Hospital admissions decreased.",
            "answer_claims": [{"component_id": "C1", "text": "Hospital admissions decreased.", "cite": ["PMID:1"]}],
            "answer_components": [component], "retrieved_chunks": [chunk()], "regen_count": 2,
        })
    assert result["answer"] == "The retrieved evidence does not establish: Fewer hospital admissions."
    assert result["answer_claims"] == []
    assert result["evidence_status"] == "insufficient"


def test_named_study_is_selected_by_identity_instead_of_leading_rank():
    from medrag.agent.nodes import grade_relevance
    wrong = chunk("2", "Unrelated study: survival was 96%.")
    target = chunk()
    llm = MagicMock()
    llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"source_ids": ["PMID:1"]})),
        MagicMock(content=json.dumps({"relevant": True, "score": 1, "components": [{
            "requirement": "Diagnostic performance", "status": "supported", "evidence_ids": ["E1"],
        }]})),
    ]
    with patch("medrag.agent.nodes.make_llm_think", return_value=llm), patch("medrag.agent.nodes.make_llm_fast", return_value=llm):
        result = grade_relevance({"query": "What did the diagnostic study find?",
                                 "source_scope": "single_study", "retrieved_chunks": [wrong, target]})
    assert result["selected_sources"] == ["PMID:1"]
    assert result["retrieved_chunks"] == [target]
    assert result["answer_components"][0]["evidence"][0]["quote"] == target.text


def test_invented_quote_and_misassigned_source_cannot_become_supported():
    for span in [
        {"chunk_id": "pubmed:1:0", "quote": "Patient survival improved."},
        {"chunk_id": "pubmed:1:0", "citation": "PMID:2", "quote": chunk().text},
    ]:
        result = bind_components([{"requirement": "Patient benefit", "status": "supported",
                                   "evidence": [span]}], [], [chunk()])
        assert result[0]["status"] == "missing"
        assert result[0]["evidence"] == []
        assert result[0]["gap"]


def test_adjacent_source_cannot_fill_a_component_or_hide_missing_comparator():
    accepted, issues = bind_claims([
        {"component_id": "C1", "text": "An unrelated review says 96%.", "cite": ["PMID:2"]},
    ], outline())
    assert not accepted
    assert any("omitted" in issue for issue in issues)


def test_missing_clinical_outcome_is_preserved_alongside_supported_diagnostics():
    status, gap = outline_status(outline())
    assert status == "partial"
    assert "prospective patient benefit" in gap.casefold()


def test_gap_rewrite_cannot_replace_the_requested_outcome():
    before = outline()
    result = repair_gaps(before, [
        {"component_id": "C1", "gap": "Wrongly overwrite the supported result"},
        {"component_id": "C2", "gap": "The study does not establish fewer hospital admissions."},
    ], ["C2"])
    assert result[0] == before[0]
    assert result[1]["gap"] == before[1]["gap"]
    assert result[1]["status"] == "missing"


def test_comparator_number_missing_from_answer_requests_component_repair():
    missing = missing_numeric_details(outline(), [{
        "component_id": "C1", "text": "Sensitivity was 92% in 30 participants; P < .001",
    }])
    assert missing == {"C1": ["92% vs. 84%"]}


def test_last_numeric_repair_quotes_only_the_bound_source():
    claims = [{"component_id": "C1", "text": "Sensitivity was 92% in 30 participants; P < .001", "cite": ["PMID:1"]}]
    repaired = restore_numeric_quotes(outline(), claims)
    assert repaired[0] == claims[0]
    assert repaired[-1]["text"] == f'The study reports: "{chunk().text}"'
    assert repaired[-1]["cite"] == ["PMID:1"]
    assert not missing_numeric_details(outline(), repaired)
    assert not any(c["component_id"] == "C2" for c in repaired)


def test_targeted_repair_retains_the_other_component():
    from medrag.agent.nodes import generate_answer_node
    components = outline()[:1]
    other = dict(components[0], id="C2", requirement="Study population")
    preserved = {"component_id": "C2", "text": "There were 30 participants", "cite": ["PMID:1"]}
    response = {"claims": [
        {"component_id": "C1", "text": "Sensitivity was 92% vs. 84%; P < .001", "cite": ["PMID:1"]},
    ], "evidence_status": "complete", "confidence": 0.5}
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(response))
    with patch("medrag.agent.nodes.make_llm_fast", return_value=llm):
        result = generate_answer_node({
            "query": "Compare performance and report the population",
            "retrieved_chunks": [chunk()], "answer_components": [*components, other],
            "answer_claims": [preserved], "repair_component_ids": ["C1"],
            "regen_count": 1, "faithfulness_issues": "C1 missing the comparator",
        })
    assert preserved in result["answer_claims"]
    assert "92% vs. 84%" in result["answer"]
    assert "30 participants" in result["answer"]


def test_development_count_cannot_be_used_as_validation_denominator():
    source = chunk(text="We developed the model using 820 images from 71 patients. Validation AUC was 0.87.")
    components = bind_components([{
        "requirement": "Model development and performance", "status": "supported",
        "evidence_ids": ["E1", "E2"], "required_details": ["Validation AUC was 0.87."],
    }], [], [source])
    wrong, issues = bind_claims([{"component_id": "C1", "text": "Validation on 820 images from 71 patients yielded AUC 0.87.", "cite": ["PMID:1"]}], components)
    assert not wrong and issues
    repaired = restore_numeric_quotes(components, wrong)
    assert any("developed the model using 820 images" in c["text"] for c in repaired)
    assert any("Validation AUC was 0.87" in c["text"] for c in repaired)
    correct, issues = bind_claims(repaired, components)
    assert correct and not issues


def test_null_comparison_and_nonnumeric_method_are_restored():
    source = chunk(text="Pressure correlated with mass (P = 0.02), whereas exposure duration did not (P = 0.7). The method combined rotated sampling with Hadamard encoding.")
    components = bind_components([{
        "requirement": "Methods and associations", "status": "supported", "evidence_ids": ["E1", "E2"],
        "required_details": ["Pressure correlated with mass (P = 0.02)"],
    }], [], [source])
    claims = [{"component_id": "C1", "text": "Pressure correlated with mass (P = 0.02). The method used sampling.", "cite": ["PMID:1"]}]
    repaired = restore_numeric_quotes(components, claims)
    assert any("duration did not (P = 0.7)" in c["text"] for c in repaired)
    assert any("Hadamard encoding" in c["text"] for c in repaired)


def test_missing_outcome_does_not_invent_absent_patient_data():
    source = chunk(text="A simulated workflow used archived examinations with histopathology as the reference.")
    components = bind_components([{
        "requirement": "Reduced invasive procedures versus usual care", "status": "missing",
        "evidence_ids": ["E1"], "gap": "There were no real patients or biopsies.",
    }], [], [source])
    assert "no real" not in components[0]["gap"]
    assert "Reduced invasive procedures versus usual care" in components[0]["gap"]
    assert repair_gaps(components, [{"component_id": "C1", "gap": "The study does not establish benefit because no patients existed."}], ["C1"]) == components


def test_method_purpose_alone_recovers_same_source_steps_not_neighboring_study():
    source = chunk(text="The method addresses slow acquisition. The technique combined rotated sampling and Hadamard encoding.")
    neighbor = chunk("2", "The technique used a different filter.")
    components = bind_components([{"requirement": "Problem addressed by the reconstruction method", "status": "supported", "evidence_ids": ["E1"]}], [], [source, neighbor])
    quotes = [s["quote"] for s in components[0]["evidence"]]
    assert "The technique combined rotated sampling and Hadamard encoding." in quotes
    assert neighbor.text not in quotes


def test_boundary_distinguishes_measured_outcome_from_requested_comparison():
    from medrag.agent.nodes import grade_relevance
    for comparison_supported in (False, True):
        source = chunk(text=("Mortality was 5% with treatment versus 10% with control among 80 randomized participants."
                             if comparison_supported else "Mortality was 5% among 80 participants in one treatment cohort."))
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=json.dumps({"assessments": [{
            "id": "C1", "outcome_measured": True, "requested_comparison_supported": comparison_supported,
            "outcome_evidence_ids": ["E1"], "comparison_evidence_ids": ["E1"] if comparison_supported else [],
            "design_evidence_ids": [],
        }]}))
        with patch("medrag.agent.nodes.make_llm_think", return_value=llm):
            result = grade_relevance({"query": "Does the study establish comparative mortality benefit?",
                                     "source_scope": "single_study", "selected_sources": ["PMID:1"],
                                     "answer_mode": "evidence_boundary", "answer_requirements": ["Comparative mortality benefit"],
                                     "retrieved_chunks": [source]})
        assert result["relevant"] is True  # A valid missing comparator needs no new study.
        assert result["answer_components"][0]["requirement"] == "Comparative mortality benefit"
        assert result["answer_components"][0]["status"] == ("supported" if comparison_supported else "missing")


def test_complete_digits_do_not_erase_who_was_assisted():
    from medrag.agent.evidence import preserve_result_context
    source = chunk(text="With model assistance, readers increased sensitivity from 80% to 92%.")
    components = bind_components([{"requirement": "Diagnostic result", "status": "supported",
                                  "evidence_ids": ["E1"], "required_details": ["80% to 92%"]}], [], [source])
    claims = [{"component_id": "C1", "text": "The model increased sensitivity from 80% to 92%.", "cite": ["PMID:1"]}]
    result = preserve_result_context(components, claims)
    assert len(result) == 1
    assert 'With model assistance, readers' in result[0]["text"]
    assert "The model increased" not in result[0]["text"]


def test_separate_study_plans_keep_global_evidence_ids():
    from medrag.agent.nodes import grade_relevance
    first = chunk(text="Method A reduced scan time.")
    second = chunk("2", "Method B reduced noise.")
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content=json.dumps({"relevant": True, "score": .9,
        "components": [{"requirement": requirement, "status": "supported", "evidence_ids": [evidence]}]}))
        for requirement, evidence in [("Method A result", "E1"), ("Method B result", "E2")]]
    with patch("medrag.agent.nodes.make_llm_think", return_value=llm):
        result = grade_relevance({"query": "Compare method A and method B", "source_scope": "multi_source",
                                 "selected_sources": ["PMID:1", "PMID:2"], "retrieved_chunks": [first, second],
                                 "source_queries": {"PMID:1": ["Method A result"], "PMID:2": ["Method B result"]}})
    assert [c["evidence"][0]["citation"] for c in result["answer_components"]] == ["PMID:1", "PMID:2"]
    assert "Method B reduced noise" not in llm.invoke.call_args_list[0].args[0][1].content
    assert "Query: Method A result" in llm.invoke.call_args_list[0].args[0][1].content
    assert "[E2]" in llm.invoke.call_args_list[1].args[0][1].content


def test_check_schema_requires_each_gap_decision():
    from medrag.agent.nodes import _check_schema
    schema = _check_schema([{"id": "C1"}, {"id": "C2"}])
    assert schema["properties"]["component_checks"]["required"] == ["C1", "C2"]


def test_design_quote_does_not_replace_causal_explanation():
    from medrag.agent.evidence import preserve_result_context
    source = chunk(text="The 2020 study was cross-sectional and used a surrogate marker.")
    components = bind_components([{"requirement": "Why causality cannot be established", "status": "supported",
                                  "evidence_ids": ["E1"], "required_details": [source.text]}], [], [source])
    claims = [{"component_id": "C1", "text": "The cross-sectional design cannot establish a causal treatment benefit.", "cite": ["PMID:1"]}]
    assert preserve_result_context(components, claims) == claims


def test_generator_recovers_actual_result_only_from_its_bound_study():
    from medrag.agent.evidence import bind_additional_evidence, preserve_result_context
    source = chunk(text="The endpoint was volume change. Volume increased by 2.3 units at 6 months; P = .02.")
    neighbor = chunk("2", "Survival improved by 30%.")
    components = bind_components([{"requirement": "Measured volume change", "status": "supported",
                                  "evidence_ids": ["E1"]}], [], [source, neighbor])
    claim = {"component_id": "C1", "text": "Volume increased.", "cite": ["PMID:1"],
             "evidence_ids": ["E2", "E3", "E999"]}
    recovered = bind_additional_evidence([claim], components, [source, neighbor])
    quotes = [s["quote"] for s in recovered[0]["evidence"]]
    assert source.text.split('. ', 1)[1] in quotes
    assert neighbor.text not in quotes
    assert any("2.3 units at 6 months" in c["text"] for c in preserve_result_context(recovered, [claim]))
    missing = [dict(components[0], status="missing")]
    assert bind_additional_evidence([claim], missing, [source, neighbor]) == missing


def test_named_identifier_is_not_crowded_out_or_matched_in_references():
    from medrag.agent.nodes import _source_cards
    wrong = [chunk(str(i), "A receptor imaging study targeting RX20.") for i in range(2, 7)]
    reference = chunk("7", "References: RX2 tumor imaging study.")
    reference.payload['section'] = 'REF'
    target = chunk(text="Preclinical RX-2 targeted imaging found specific uptake.")
    assert list(_source_cards("What did the RX2 imaging study find?", [*wrong, reference, target])) == ['PMID:1']
    assert not _source_cards("What did the RX3 imaging study find?", [*wrong, reference, target])


def test_open_boundary_names_missing_effect_without_invented_explanation():
    source = chunk(text="The workflow was simulated using patient records and biopsy results.")
    raw = {"requirement": "Identify the real-world clinical effect that remains untested",
           "status": "missing", "evidence_ids": ["E1"],
           "missing_outcome": "prospective deployment effects on patient care"}
    components = bind_components([raw], [], [source])
    assert components[0]["gap"] == "The retrieved evidence does not establish: prospective deployment effects on patient care."
    raw['missing_outcome'] = 'patient benefit because there were no patients'
    assert 'no patients' not in bind_components([raw], [], [source])[0]['gap']


def test_exact_population_quote_does_not_inherit_another_sentences_role():
    source = chunk(text="The workflow was simulated on 40 examinations. The 40 examinations were used for testing.")
    components = bind_components([{"requirement": "Study workflow", "status": "supported",
                                  "evidence_ids": ["E1", "E2"]}], [], [source])
    claim = {'component_id': 'C1', 'cite': ['PMID:1'],
             'text': 'The study reports: "The workflow was simulated on 40 examinations."'}
    accepted, issues = bind_claims([claim], components)
    assert accepted == [claim] and not issues
