# p11fix — why the lesson hook never fired, and the evaluation arm that fixes it

Cause (counts from the p11 transcripts, replayed through completion_hook.check):
- The flag is ruled out: the p11 ON runs armed the hook with `--pre-claim-hook`, which calls
  `maybe_lesson(..., force=True)`; the config flag is never read. Summaries show pre_claim_hook=true.
- Sonnet ON: 30 tasks, every one did the work with tool calls (0 tasks with zero calls) and touched every
  numbered item. check() returned no_tool=False, numbered=[] on 30/30, so no lesson. It passed 30/30
  without help; there was no mistake for the hook to catch.
- llama ON runs: the checks look at whether a call TOUCHED an item, not whether it WORKED. Failed tasks
  were 17/30 and 18/30; check() tripped on only 1 of them in each run (numbered n10). The rest were
  wrong-content writes, failed/repeated calls, claims after errors, or no final message (the hook runs
  only at a final message): every item path was in some successful call, so nothing tripped.
  The 1 lesson event in llama_on_v2 / num_on is that n10 case.
Fix: the evaluation harness carries the lesson on the task (`task["lesson"]`, CLI `--task-lessons FILE`)
and shows it at the first completion claim with no check deciding. The transcript has a "hook" record
(fired, text, reason, at_call). Live code is unchanged; with no arm armed there is no "hook" key.
Fires when: the model sends a message without a tool call and the task has a lesson text.
Does not fire when: the model never sends such a message (step cap or transport error), or the task has
no lesson text (reason recorded).
