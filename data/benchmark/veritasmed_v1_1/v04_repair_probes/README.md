# Retained repair debugging probes

Probes 1-6 use questions.jsonl (the six original candidate failures); probe 7 uses
probe7_questions.jsonl (four failures from the first complete repair round).
Every original output is preserved and excluded from the final independent run metrics.
Each patch is relative to the code_commit recorded in its raw output.

Probe 7 metadata correction: its inherited runner recorded fast_temperature=0.2,
but the saved llms.py patch used 0.0 for structured calls. The raw artifact is not
rewritten. Subsequent formal runs obtain settings directly from the actual model objects.

The first independent repair round is retained in ../v04_repair_round1/. Its complete
development run passed 11/15, with focus results 4/6 and 5/6. It exposed four failures:
missing actual results (002/006), wrong source identity (010), and a generic gap (013).
The original first v0.4 candidate artifacts remain separate and unchanged.
