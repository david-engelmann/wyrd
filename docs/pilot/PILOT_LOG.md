# RAG-001 pilot log

Pilot purpose, per the charter: replace the section 8 planning
assumptions with measurements before the full RAG-001 spec seals. Pilot
model is claude-haiku-5-5 through the Anthropic window helper. All
attempts across all runs are logged in the attempts ledgers named below
and summarised in ../SPEND_LOG.md.

## Run 1 (2026-11-03): total failure, driver environment bug

100 attempts, all HTTP 401 authentication errors. The driver passed a
stripped environment to the helper subprocess, cutting off credential
resolution. The hand-run probe with the full environment had succeeded,
so the defect only appeared at scale. The driver also lacked an
identical-failure halt and logged-and-continued through all 100.
Retrieval reads from run 1 remain valid (retrieval is local): poison
documents reached the top 5 for all 20 questions at every dose of 1 or
more, gold retrieval 20 of 20 clean and at sham, 17 of 20 at dose 5.
Ledger: attempts_run1_failed401.jsonl, results_run1_failed401.json.

## Run 2 (2026-11-03): completed, generation results void

100 attempts, zero failures, usage 26,566 input and 3,867 output tokens.
Headline trap: poisoned arms produced empty answers on 57 of 60 calls
and never stated the poison answer, which looked like heroic resistance.
It was an artifact. The helper prints only text blocks, and on
conflicted prompts the model spends its output budget on thinking
blocks first. At max_tokens 60 the budget died before any text. A manual
reproduction at max_tokens 60 returned empty stdout with 60 output
tokens used; the same prompt at 1024 returned a real answer in 106
output tokens. Run 2 generation numbers are void and will not be
reported as results anywhere. Its retrieval reads stand and agree with
run 1.

Lesson recorded for the full spec: max_tokens is an experimental
variable, empty output is its own outcome category (no text within
budget), and token budgets must be set from the conflicted arm's usage,
never from the clean arm's. Clean calls here used about 6 output tokens;
conflicted calls needed about 106.

## Run 3 (2026-11-03): corrected budget, the valid run

Same corpus and conditions, max_tokens 1024, driver halt on three
consecutive failures, environment inherited with NO_PROXY overridden
only. 100 attempts, zero failures, usage 26,566 input and 13,576 output
tokens. Full analysis with hand labels in ANALYSIS.md. Headlines:
retrieval is won at every dose (poison in top 5 for all 20 questions,
mean 4.15 poison documents at dose 5); raw containment success reads
0/6/16 of 20 at doses 1/2/5 but hand labelling shows the model mostly
discusses the planted claim to reject it; final-assertion success is
0/0/1 of 20, lead-assertion success 0/0/5 of 20; sham is inert at 20 of
20 correct. The full spec inherits an assertion-based primary judge,
three reported reads, and budgets set from the conflicted arm.
