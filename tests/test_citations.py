"""Evidence validation at the model-output boundary."""

from medrag.agent.utils import build_answer_from_claims, validate_citations


def test_malformed_claims_are_ignored_without_mutating_input():
    claims = [
        None,
        {"text": 42, "cite": ["PMID:1"]},
        {"text": "Unsupported cite", "cite": [17]},
        {"text": "Cite is not a list", "cite": "PMID:1"},
        {"text": "Supported", "cite": ["PMID:1"]},
    ]
    original = [None, {"text": 42, "cite": ["PMID:1"]},
                {"text": "Unsupported cite", "cite": [17]},
                {"text": "Cite is not a list", "cite": "PMID:1"},
                {"text": "Supported", "cite": ["PMID:1"]}]

    validated = validate_citations(claims, [{"citation": "PMID:1"}])

    assert validated == [{"text": "Supported", "cite": ["PMID:1"]}]
    assert claims == original


def test_bracketed_citations_are_normalized_and_rendered_once():
    claims = [{"text": "First.", "cite": ["[PMID:1]", "PMID:1", " [PMC:2] "]},
              {"text": "Second.", "cite": ["[PMID:1]"]}]
    validated = validate_citations(
        claims, [{"citation": "PMID:1"}, {"citation": "PMC:2"}]
    )

    assert validated == [
        {"text": "First.", "cite": ["PMID:1", "PMC:2"]},
        {"text": "Second.", "cite": ["PMID:1"]},
    ]
    answer, citations = build_answer_from_claims(validated)
    assert answer == "First [PMID:1] [PMC:2]. Second [PMID:1]."
    assert citations == ["PMID:1", "PMC:2"]
    assert claims[0]["cite"][0] == "[PMID:1]"
