# Prompt changes in this release

## `braindiff/dataset/report_rewrite_prompt.py` — prior-report edit prompt (`BaselineEditSystemPrompt`)

The edited prior reports used to train and evaluate BrainDiff were produced with the original version of this prompt. The version shipped
here differs from it for privacy:

- **Worked example Case 13** (input and output) was replaced with a synthetic example of the same structure: a findings-only report with
  normal-anatomy statements, extracranial content and generic negatives that the rules delete, and two abnormal findings that they keep.
  The original example was taken from a real patient report.
- **Worked example Case 15** (input and output) was replaced with a synthetic example of the same structure: a current and a prior
  measurement, a comparison-only clause, a change-only sentence, a resolved finding and a numbered impression.
- **One example sentence under `<change_rules>`** (the "delete the sentence (only the change is described ...)" rule) was replaced with the
  synthetic change-only sentence used in the new Case 15. The original sentence matched real reports.

An internal annotation tag (`[user, 2026-10-30]`) was also removed from one rule in both this prompt and the follow-up edit prompt
(`FollowupEditSystemPrompt`); it was part of the text sent to the model but carried no instruction.

The instructions, rules, output format, validation checklist and every other example are unchanged. Because the examples differ, running
this prompt will not reproduce the shipped training targets exactly.

How this was checked: every sentence of eight or more words in each LLM prompt of the project was compared against the report corpora the
project used. Only this prompt contained text from a real report. In the shipped version, every sentence that also occurs in a real report
is generic radiology phrasing that occurs in five or more reports.
