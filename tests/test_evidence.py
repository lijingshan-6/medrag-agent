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
    assert result[0]["required_details"] == ["We enrolled 31 patients and 29 controls."]
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
    assert "prospective patient benefit" in gap


def test_targeted_gap_can_be_clarified_without_changing_supported_results():
    before = outline()
    result = repair_gaps(before, [
        {"component_id": "C1", "gap": "Wrongly overwrite the supported result"},
        {"component_id": "C2", "gap": "The study does not establish fewer hospital admissions."},
    ], ["C2"])
    assert result[0] == before[0]
    assert result[1]["gap"] == "The study does not establish fewer hospital admissions."
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
