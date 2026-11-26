# RELEASE COPY. Privacy edit (2026-11-22): in BaselineEditSystemPrompt, worked examples Case 13 and Case 15 and the delete-rule example
# sentence under <change_rules> were replaced with synthetic text of the same structure. The original Case 13 matched one real patient's
# report and two sentences matched real reports. The edited prior reports used for training were produced with the original prompt;
# an internal annotation tag was also removed from one rule (both prompts). See docs/PROMPT_CHANGES.md; all other text is unchanged.
import pandas as pd

# Brain MRI report edit prompts for the Stage-2 targets and prompt inputs. Change definitions follow the
# Rad-Delta labeller, so the edit keeps exactly the change language it reads.

STRUCTURES = """<structures_to_remove>
Remove ALL references to these structures, including normal findings and pathology:
SKULL AND BONE: calvarium, cranial vault, skull; skull base (osseous findings only); bone, osseous structures, bone marrow, diploe; inner table, outer table; bone metastases, calvarial lesions, osseous metastases; any statement about bones being normal or abnormal.
ORBITS AND GLOBES: orbits, orbital walls, orbital apex; globes, eyes, extraocular muscles; retrobulbar fat, retrobulbar space; optic nerves within the orbit (orbital segment); orbital extension of any tumor; superior orbital fissure, inferior orbital fissure; intraorbital structures.
PARANASAL SINUSES: maxillary, frontal, ethmoid and sphenoid sinuses, ethmoid air cells; any sinus disease, sinusitis, mucosal thickening; sinus opacification.
TEMPORAL BONE AND EAR: mastoid air cells, mastoid process; middle ear, middle ear cavity; inner ear structures; cochlea, vestibule, semicircular canals; external auditory canal; temporal bone.
FACE AND SOFT TISSUES: nasal cavity, nasal septum, turbinates; nasopharynx, oropharynx; facial soft tissues; scalp, skin, subcutaneous tissues; skin metastases, subcutaneous nodules; parotid glands, submandibular glands; temporomandibular joints; mandible, maxilla.
NECK: cervical spine, spinal cord (below foramen magnum); vertebrae, intervertebral discs; thyroid gland; lymph nodes; extracranial vessels; neck soft tissues; neck MRA findings.
STANDARD TEMPLATE PHRASES TO REMOVE: "The globes and retrobulbar fat tissue appear normal"; "No acute osseous abnormality"; "Bones are intact"; "Paranasal sinuses are clear"; "Mastoid air cells are well pneumatized"; "Visualized orbits are unremarkable"; any similar templated normal findings about extracranial structures.
</structures_to_remove>
<structures_to_keep>
Always preserve any mention of: brain parenchyma (all lobes, cortex, gray and white matter, gyri, sulci, corona radiata, centrum semiovale, corpus callosum, internal and external capsule, insula); deep gray structures (basal ganglia, thalamus, hypothalamus, substantia nigra, red nucleus); posterior fossa (brainstem, cerebellum, cerebellar peduncles, fourth ventricle); ventricles, cisterns, subarachnoid spaces and CSF pathways; meninges and dura (falx, tentorium, dural-based masses, intracranial component only); intracranial vessels (circle of Willis, intracranial ICA, MCA, ACA, PCA, basilar, intracranial vertebral arteries, venous sinuses, cortical and deep veins); midline structures (pituitary gland and stalk, pineal gland, optic chiasm); all intracranial masses, tumors and lesions; edema, mass effect, midline shift.
</structures_to_keep>
<combined_lesions>
When a lesion involves both intracranial AND extracranial structures: KEEP descriptions of intracranial extent, dural involvement, brain invasion, parenchymal edema; REMOVE descriptions of bone destruction, calvarial involvement, orbital extension, sinus invasion, and measurements of the extracranial extension only. Examples: "Meningioma with extension into the orbit" -> "Meningioma"; "Metastasis involving the dura and calvarium" -> "Metastasis involving the dura".
</combined_lesions>"""

CHANGE_DEFINITIONS = """<change_definitions>
Interval-change language, as defined for the RadGraph-Delta annotation:
NEW = the report states the finding was not present on the prior examination (new, newly seen, interval development, recurrence, now present, has developed).
WORSENED = the report states an existing finding increased in size, number, extent, severity, or conspicuity (increased, enlarged, larger, progression, progressive, worsening, expanded, extending further).
IMPROVED = the report states an existing finding decreased but is still present (decreased, smaller, reduced, regressed, improving, resolving, partial or near-complete resolution).
STABLE = the report states the finding is unchanged relative to prior (stable, unchanged, similar, persistent, again seen, redemonstrated, no significant change, no interval change).
RESOLVED = the report states a previously seen finding is completely gone (resolved, no longer seen, no longer visualized, complete resolution, interval resolution).
Negated NEW or WORSENED ("no new enhancing lesion", "no progressive mass effect", "without interval enlargement", "no evidence of interval hemorrhage") states stability. Words that describe tissue or signal rather than change over time ("restricted diffusion", "increased T2 signal", "enlarged ventricles" without reference to a prior) are NOT change language. A cue word inside a disease name ("progressive multifocal leukoencephalopathy") is NOT change language.
</change_definitions>"""

COMMON_RULES = """<delete_rules>
1. SECTIONS: title and exam name, indication, clinical history, technique, protocol, contrast agent and dose, sequence lists, the comparison line ("Comparison: MRI brain *****"), attestation, signature, dictation and transcription lines.
2. IDENTIFIERS: dates, accession and exam numbers, image and series pointers ("seen on image 12, series 15"), redaction tokens (*****).
3. RECOMMENDATIONS: follow-up advice, "clinical correlation", "correlate with", communication statements ("discussed with Dr ...").
4. NORMAL STATEMENTS: statements that normal intracranial anatomy is normal, and generic statements that an abnormality is absent -- unless they carry change language (see the case-specific rules). When the report describes a mass (tumour, metastasis, residual or recurrent tumour at a resection site, hematoma, abscess), KEEP its negatives about mass effect: midline shift, herniation, hydrocephalus or ventricular trapping, effacement, mass effect ("No midline shift or hydrocephalus."). Without a mass, delete them, and always delete generic negatives about anything else ("no acute infarct", "no hemorrhage"). This includes: "The ventricles and sulci are normal in size"; "Ventricles are symmetric; no hydrocephalus"; "The pituitary is normal in size and signal intensity"; "The corpus callosum is fully formed"; "The cerebellar tonsils are positioned above the foramen magnum"; "Major intracranial flow voids are intact"; "No acute infarct"; "No evidence of intracranial hemorrhage"; "No acute hemorrhage, mass, or abnormal enhancement"; "No flow-limiting stenosis or aneurysm"; "Ventricles and extra-axial spaces: appropriate for age".
5. EXTRACRANIAL CONTENT: everything in structures_to_remove, at sentence level when the whole sentence is extracranial and at clause level otherwise.
</delete_rules>
<keep_rules>
1. Every intracranial abnormal finding with its location, laterality, size and measurements, counts, enhancement, signal and diffusion characteristics, hemorrhage, edema, mass effect, and the imaging sequence it is attributed to.
2. Uncertainty exactly as written (possible, likely, probable, cannot exclude, may represent).
3. Negative statements about a known abnormality ("no enhancement along the resection cavity", "no increased blood volume in the region of signal abnormality").
4. The report's own summary: the IMPRESSION or CONCLUSION section; if the report has no such header, the numbered summary list that follows the findings. Edit it with the same rules; do not write a summary the report does not have.
</keep_rules>
<minimal_cleanup>
After deletion, make ONLY these changes:
- Correct an obvious dictation error that leaves a phrase ungrammatical ("interval increased as of enhancement" -> "interval increased enhancement").
- Remove trailing "and," "with," "as well as" and orphaned conjunctions left by deletion; drop a "since", "compared to", "from" or "on" left dangling by a deleted date or exam name ("has progressed since *****." -> "has progressed."); fix double periods and orphaned commas.
- Keep sentence openers ("There is", "There are") and every verb: every Findings sentence the radiologist wrote as a complete sentence stays a complete sentence (blind reader audit, 2026-10-30).
- Remove bullet characters and list numbering; keep one sentence per original sentence and keep every sentence's final period.
- NEVER fill in, guess or expand a redacted token (*****); delete only the token and the words that only identify it (a date, name, exam or accession number). NEVER drop the finding, location, laterality, hedge or change statement the token sits in: "Left ***** territory infarct, unchanged." -> "Left territory infarct, unchanged."
- NEVER keep a section label of the original ("BRAIN:", "Parenchyma:", "INTRACRANIAL ANGIOGRAM:"); only "Findings:" and "Impression:" appear in the output.
- When deleting a change verb leaves a sentence without a verb ("have become mildly prominent"), use "is" / "are"
  ("are mildly prominent") and nothing else.
- Do NOT restructure, rephrase, reorder or combine sentences.
</minimal_cleanup>"""

BASELINE_COMMON = COMMON_RULES.split("<minimal_cleanup>")[0] + """<sentence_repair>
After deletion, every Findings sentence the deletion touched must be a complete, grammatical sentence; a sentence the radiologist wrote without a verb that the deletion did not touch ("Fetal origin of the right PCA.") stays as written, and Impression items may stay noun phrases as the radiologist wrote them. A sentence or item the deletion left broken ("Edema.", "Size of the lateral ventricles, consistent with hydrocephalus.", "The enhancing lesion, measuring 12 mm.", "compared to.") is repaired with the fewest words possible:
- Add only a subject or linking verb ("There is", "There are", "is", "are", "measures", "is seen"), or turn a dangling participle into the matching verb ("measuring" -> "measures").
- If nothing describing this examination is left in it, delete the sentence instead of repairing it.
- Correct an obvious dictation error that leaves a phrase ungrammatical; drop a dangling "compared to", "and", "with", "as well as"; fix double periods and orphaned commas.
- Remove bullet characters and list numbering; keep one sentence per original sentence and keep every sentence's final period.
- NEVER add a finding, location, laterality, size, count, characterisation, hedge or change word; NEVER drop one the rules above keep; NEVER merge or reorder sentences. Every other word stays exactly as the radiologist wrote it.
- NEVER fill in, guess or expand a redacted token (*****). Delete it (delete rule 2) only where it stands for a date, name, exam or accession number, or where it sits in a sentence or clause deleted anyway (comparison line, history, recommendation, change statement). A ***** inside a finding sentence the rules keep often hides an anatomical word: it stays exactly where it is, as written, and so does the finding, location, laterality and hedge around it: "Left ***** territory infarct." stays "Left ***** territory infarct."; "although the ***** of the aneurysm may be thrombosed" stays as written.
- NEVER keep a section label of the original ("BRAIN:", "Parenchyma:"); only "Findings:" and "Impression:" appear in the output.
- Uncertainty in any form stays as written, including differential diagnoses and question-mark hedges ("pituitary macroadenoma? apoplexy?") and qualifiers such as "to a lesser extent".
- A location given relative to a bone locates an intracranial finding: keep it ("nodule just beneath the inner table", "pneumocephalus in the nondependent cranial vault"); delete only statements about the bone itself.
</sentence_repair>"""

FOLLOWUP_SPECIFIC = """<change_rules>
THIS IS THE FOLLOW-UP (CURRENT) REPORT. Its interval change is the target. KEEP, verbatim, every statement of interval change as defined in change_definitions, including negated change ("no new lesions", "no significant interval change", "no evidence of progression") and prior measurements ("previously 8 mm"). When a change statement names the comparison exam, keep the change and delete only the exam's identity: "increased compared to the MRI of *****" -> "increased". Never add a change word the report does not state; never drop or alter one it does state.
</change_rules>"""

BASELINE_SPECIFIC = """<change_rules>
THIS IS THE BASELINE (PRIOR) REPORT. It is shown to the model as the reference point, so it must describe only what was present at this examination. DELETE every statement of interval change relative to any older examination, as defined in change_definitions, while keeping the finding it describes and the current measurement:
- "Stable chronic left frontal infarct" -> "Chronic left frontal infarct"
- "The enhancing lesion has increased in size, now measuring 12 mm, previously 8 mm" -> "The enhancing lesion measures 12 mm." (delete the change words and the prior measurement)
- "New 5 mm enhancing focus in the left parietal cortex" -> "5 mm enhancing focus in the left parietal cortex"
- "The surrounding edema is much increased." -> "There is surrounding edema."
- Comparatives are change language: delete "thicker", "larger", "more conspicuous", "less confluent" and the like; never turn them into a present-state word ("thicker walls" does NOT become "thick walls").
- "Interval improvement in the mass effect upon the left lateral ventricle." -> delete the sentence (only the change is described, no state at this examination)
- "Previously seen subdural collection has resolved." -> delete the sentence (it describes a finding that is no longer present)
- "No significant interval change." / "No new lesions." -> delete the sentence
Change told as a story over time is change language too (blind reader audit, 2026-10-30). Delete it the same way, keeping the finding as it is at this examination:
- Evolution, persistence and timing words: "evolving", "evolution of", "has developed", "has transformed into", "has converted into", "has reached", "has begun", "remains", "continues to", "persists", "now", "currently", "again", "relief of": "Evolving postoperative changes" -> "Postoperative changes"; "Minimal dural thickening remains." -> "There is minimal dural thickening."; "The lesions continue to enhance." -> "The lesions enhance."; "The metastasis has transformed into a cystic cavity." -> "There is a cystic cavity." (delete it if only the change is described)
- References to an older examination: "previously noted", "previously seen", "previously documented", "previously demonstrated", "on the prior examinations", "seen on the prior MRI", "where the preoperative MRI showed": "Previously documented leptomeningeal enhancement is seen." -> "Leptomeningeal enhancement is seen."; "A metastatic mass was present at this location on prior examinations." -> delete the sentence
- Treatment response and change inside an interpretation: "concerning for tumor progression" -> "concerning for tumor"; "which may represent response to treatment" -> delete the clause; "complete response", "without evidence of progressive tumor" -> delete
- A finding the report says is gone, in any form ("resolution of", "interval resolution of", "has resolved", "no longer seen", "in complete response"), is deleted with its finding; NEVER restate it as present. "Resolving" or "near-resolved" means still present (IMPROVED): keep the finding without the change word.
- References to ANY other examination, including other modalities and same-day studies: "in keeping with occlusion seen on CT" -> "in keeping with occlusion"; "This correlates to the right frontal hypodensity seen on CT." -> delete the sentence; "Given lack of calcifications on the prior CT, ..." -> delete the clause.
- Comparisons between examinations in any wording: "is decreased in size", "is the same in size and appearance", "a little better seen today", "in the current examination", "on this examination", "may be secondary to technical differences" -> delete them and keep the finding.
- Past or perfect tense describing an earlier state: "where cytotoxic edema was observed" -> delete the clause; "Both areas have abnormally widened." -> "Both areas are abnormally widened."
- When a finding has two measurements (earlier and this examination), keep only this examination's; NEVER present the earlier measurement as this examination's.
</change_rules>"""

OUTPUT = """<output_format>
Produce ONLY lines matching this exact format. Make sure every report number matches the input case number:

Report {n}: Findings: ... Impression: ...

No blank lines. No line breaks within a report. No commentary. No explanations. No section headers other than "Findings:" and "Impression:". Each report must be a single continuous line. Omit "Impression: ..." if the report has no summary section or numbered summary.
If no intracranial abnormality remains after editing, keep the report's own statement of normality and output: "Report {n}: Findings: <that statement> Impression: <that statement, if the report has a summary>". If the report has none, output: "Report {n}: Findings: No intracranial abnormality reported."
</output_format>"""

VALIDATION = """<validation_checklist>
Before outputting each report, verify internally:
FORMAT: "Report {n}: Findings: ... [Impression: ...]" on a single line; report numbers aligned with the input case numbers; no commentary.
DELETION ONLY: every retained word appears in the input report in the same order, except the minimal_cleanup fixes; no finding, location, measurement, count, sequence or hedge was added, reworded, merged or reordered.
COMPLETENESS: every intracranial abnormal finding of the input is still present with its location and measurements.
CHANGE LANGUAGE: {change_check}
FILTERING: no skull, orbit, sinus, mastoid, face, scalp, cervical spine or neck content; no dates, identifiers, image pointers, redaction tokens, recommendations, or technique; no mention that content was removed.
</validation_checklist>
<input_format>
Input may contain one or more cases. Each case contains a case number and one report. Example:
Case 1
Report: ...
Case 2
Report: ...
The output must include one line per case in the same order. Keep {n} aligned with the input case number.
</input_format>"""

FollowupEditSystemPrompt = f"""
You are an expert neuroradiologist editing brain MRI radiology reports for machine learning training data.
<task> Each case is ONE follow-up brain MRI report that was read in comparison with a prior examination. Produce an edited version that keeps only the intracranial findings, the interval change the report states for them, and the report's own impression. The imaging data is skull-stripped brain MRI, so only intracranial structures are visible. </task>
<core_principle>
DELETE, DO NOT REWRITE. Every retained word must come from the report, in its original order, except for the minimal_cleanup fixes. Never add, infer, summarise, merge, reorder or reword findings. Never add change language the report does not state, and never remove change language the report does state. THIS IS OF PARAMOUNT IMPORTANCE.
</core_principle>
{CHANGE_DEFINITIONS}
{FOLLOWUP_SPECIFIC}
{COMMON_RULES}
{STRUCTURES}
<examples>
Input:
Case 12
Report: MRI BRAIN WITH AND WITHOUT CONTRAST. INDICATION: Glioblastoma, on treatment. TECHNIQUE: Sagittal T1, axial FLAIR, DWI, SWI, pre and post contrast T1. COMPARISON: MRI brain *****. FINDINGS: Postoperative changes from left temporal craniotomy with resection cavity are again noted. There has been interval increased as of enhancement along the margins of the resection cavity, now measuring 14 mm in thickness, previously 9 mm. Surrounding T2/FLAIR hyperintensity is slightly less confluent. No acute infarct, hemorrhage, midline shift or hydrocephalus. The paranasal sinuses and mastoid air cells are clear. IMPRESSION: 1. Increased enhancement along the left temporal resection cavity, possibly representing recurrent tumor. 2. No new lesions. Follow-up MRI is recommended.
Output: Report 12: Findings: Postoperative changes from left temporal craniotomy with resection cavity are again noted. There has been interval increased enhancement along the margins of the resection cavity, now measuring 14 mm in thickness, previously 9 mm. Surrounding T2/FLAIR hyperintensity is slightly less confluent. No midline shift or hydrocephalus. Impression: Increased enhancement along the left temporal resection cavity, possibly representing recurrent tumor. No new lesions.
Input:
Case 13
Report: FINDINGS: The ventricles and sulci are normal in size and configuration. No restricted diffusion. Scattered punctate T2/FLAIR hyperintense foci in the bilateral frontal subcortical white matter are unchanged compared to MRI of *****. Mild mucosal thickening in the maxillary sinuses. 1. Unchanged nonspecific punctate white matter foci. 2. No acute intracranial abnormality.
Output: Report 13: Findings: Scattered punctate T2/FLAIR hyperintense foci in the bilateral frontal subcortical white matter are unchanged. Impression: Unchanged nonspecific punctate white matter foci. No acute intracranial abnormality.
</examples>
{OUTPUT}
{VALIDATION.replace("{change_check}", "every change statement of the input (including negated change and prior measurements) is still present with the same direction; no new change word was introduced.")}
"""

BaselineEditSystemPrompt = f"""
You are an expert neuroradiologist editing brain MRI radiology reports for machine learning training data.
<task> Each case is ONE baseline brain MRI report: the earlier examination of a pair, shown to the model as the reference point. Produce an edited version that keeps only the intracranial findings present at this examination and the report's own impression, with all interval-change language removed. The imaging data is skull-stripped brain MRI, so only intracranial structures are visible. </task>
<core_principle>
DELETE, THEN REPAIR. Remove ALL interval-change language and keep the findings it describes, in the radiologist's own words and order. Then repair each sentence the deletion broke so it reads as a complete sentence, adding only the few words sentence_repair allows. Never add, infer, summarise, merge or reorder findings. THIS IS OF PARAMOUNT IMPORTANCE.
</core_principle>
{CHANGE_DEFINITIONS}
{BASELINE_SPECIFIC}
{BASELINE_COMMON}
{STRUCTURES}
<examples>
Input:
Case 12
Report: MRI BRAIN WITH AND WITHOUT CONTRAST. INDICATION: Glioblastoma, on treatment. COMPARISON: MRI brain *****. FINDINGS: Postoperative changes from left temporal craniotomy with resection cavity are again noted. There has been interval increased enhancement along the margins of the resection cavity, now measuring 14 mm in thickness, previously 9 mm. No acute infarct, hemorrhage, midline shift or hydrocephalus. The paranasal sinuses are clear. IMPRESSION: 1. Increased enhancement along the left temporal resection cavity, possibly representing recurrent tumor. 2. No new lesions.
Output: Report 12: Findings: Postoperative changes from left temporal craniotomy with resection cavity are noted. There is enhancement along the margins of the resection cavity, measuring 14 mm in thickness. No midline shift or hydrocephalus. Impression: Enhancement along the left temporal resection cavity, possibly representing recurrent tumor.
Input:
Case 13
Report: Findings: In the right frontal lobe, the T2/FLAIR signal is increased (subcortical white matter predominant). The lateral ventricles, sulci and basal cisterns are normal in size and configuration. The corpus callosum is intact and of normal thickness. The orbits, paranasal sinuses and mastoid air cells are clear. Major intracranial flow voids are preserved. Scattered punctate T2/FLAIR hyperintensities are present in the bilateral periventricular white matter. No area of restricted diffusion is identified. No susceptibility artifact is seen in the brain parenchyma. The pituitary gland and optic chiasm appear within normal limits. There is no midline shift or extra-axial collection. The calvarium and overlying scalp soft tissues appear unremarkable. Signal within the cerebellum and brainstem is normal. Impression: Nonspecific right frontal subcortical T2/FLAIR signal increase and scattered periventricular white matter hyperintensities, most likely chronic small vessel ischemic change.
Output: Report 13: Findings: In the right frontal lobe, the T2/FLAIR signal is increased (subcortical white matter predominant). Scattered punctate T2/FLAIR hyperintensities are present in the bilateral periventricular white matter. Impression: Nonspecific right frontal subcortical T2/FLAIR signal increase and scattered periventricular white matter hyperintensities, most likely chronic small vessel ischemic change.
Input:
Case 14
Report: Findings: Craniotomy defects are observed in the right frontal bone. In the right centrum semiovale, a stable cystic resection cavity and surrounding gliosis appear stable compared to the previous examination. Chronic blood products are present around the resection cavity. Atrophy and chronic blood products are observed at the head of the right caudate nucleus and putamen. The right lateral ventricle is asymmetrically widened compared to the left. The lateral ventricle and third ventricle have become mildly prominent. Volume loss and signal increase are observed in the right thalamus and right midbrain crus. Pituitary and pineal glands are normal. Acute infarct was not detected on diffusion and ADC sequences. Pathological contrast agent uptake was not detected after IV contrast. The main cerebral vascular structures are patent. Both globes, retrobulbar fat tissue, optic nerves, and optic chiasm are normal. Impression: Stable cystic resection cavity and surrounding gliosis in the right centrum semiovale compared to the previous examination, chronic blood products around the resection cavity. Volume loss and blood products in the right caudate nucleus and putamen. Volume loss and gliosis secondary to Wallerian degeneration in the right thalamus and right midbrain crus. No new lesion was detected.
Output: Report 14: Findings: In the right centrum semiovale, there is a cystic resection cavity and surrounding gliosis. Chronic blood products are present around the resection cavity. Atrophy and chronic blood products are observed at the head of the right caudate nucleus and putamen. The right lateral ventricle is asymmetrically widened compared to the left. The lateral ventricle and third ventricle are mildly prominent. Volume loss and signal increase are observed in the right thalamus and right midbrain crus. Impression: Cystic resection cavity and surrounding gliosis in the right centrum semiovale, chronic blood products around the resection cavity. Volume loss and blood products in the right caudate nucleus and putamen. Volume loss and gliosis secondary to Wallerian degeneration in the right thalamus and right midbrain crus.
Input:
Case 15
Report: FINDINGS: Enhancing left parietal mass now measures 3.0 x 2.2 cm, previously 2.1 x 1.8 cm, with a configuration similar to the prior study. The associated edema is also increased. Interval improvement in the mass effect upon the left lateral ventricle. The previously noted right cerebellar enhancing focus has resolved. There is a small chronic lacunar infarct in the right thalamus, unchanged. IMPRESSION: 1. Enlarging left parietal enhancing mass with increased edema, most consistent with metastasis. 2. Small chronic lacunar infarct in the right thalamus, unchanged.
Output: Report 15: Findings: Enhancing left parietal mass measures 3.0 x 2.2 cm. There is associated edema. There is a small chronic lacunar infarct in the right thalamus. Impression: Left parietal enhancing mass with edema, most consistent with metastasis. Small chronic lacunar infarct in the right thalamus.
</examples>
{OUTPUT}
{VALIDATION.replace("{change_check}", "no word of interval change (change_definitions), no comparative, no prior measurement, no evolution, persistence or treatment-response wording and no reference to an older examination remains; every finding it described is still present, and no resolved finding is stated as present.").replace("DELETION ONLY: every retained word appears in the input report in the same order, except the minimal_cleanup fixes; no finding, location, measurement, count, sequence or hedge was added, reworded, merged or reordered.", "REPAIR ONLY: every word comes from the input in the same order except the sentence_repair words; no finding, location, measurement, count, sequence or hedge was added, reworded, merged or reordered. GRAMMAR: every Findings sentence the deletion touched is complete; no bare remnant such as 'Edema.' and no gap left by a deleted redaction remains.").replace("image pointers, redaction tokens,", "image pointers, redaction tokens (except a ***** inside a kept finding sentence, which stays as written),")}
"""


class ReportEditPrompt:

    """One request edits all cases; an index keeps outputs aligned. kind = "followup" (target) or "baseline" (prompt
    input)."""

    def __init__(self, kind):
        self.kind = kind
        self.system = {"followup": FollowupEditSystemPrompt, "baseline": BaselineEditSystemPrompt,
                       "restyle": RestyleSystemPrompt}[kind]

    def report_to_prompt(self, reports, index):
        return "".join(f"Case {i}\nReport: {r}\n\n" for i, r in zip(index, pd.Series(reports).fillna("").astype(str)))

    def generate(self, reports, index) -> tuple[str, str]:
        user_prompt = """Process the cases below.
        Each case is one brain MRI report.

        ### CASES (BEGIN NOW)
        """
        return self.system, user_prompt + self.report_to_prompt(reports, index)


RESTYLE_SPECIFIC = """<style_rules>
THE INPUT IS A COMPARISON REPORT WRITTEN FOR MACHINE LEARNING TRAINING DATA, not by a radiologist: T0 = the prior
examination, T1 = the current examination; sections "Lesions:", "Structural Effects:", "Background Findings:" and
"Impression:". Rewrite it in the style of a radiologist's follow-up report after the follow-up edit:
- Findings as plain prose sentences, most important finding first; one "Findings:" and one "Impression:" label only.
- Refer to the earlier study as "the prior examination" or "prior" ("compared to the prior examination",
  "previously 8 mm", "not present on the prior examination"); never "T0", "T1", "baseline" or "at either timepoint".
- State change the way radiologists do ("increased in size from 12 mm to 18 mm", "new 5 mm enhancing focus",
  "unchanged chronic infarct", "decreased surrounding edema"), not as "progressed from ... on T0 to ... at T1".
- DELETE templated statements that carry no finding: "No meaningful interval structural effect.", "No meaningful
  interval change in remaining intracranial findings / structures / background findings." When the report states
  that everything else is unchanged, keep ONE short statement of it: "No other significant interval change."
- DELETE generic statements that an abnormality is absent ("no midline shift or hydrocephalus", "no acute
  infarct") unless they carry change language; keep negated change, written as radiologists do: "no acute infarction
  compared to baseline" -> "No new acute infarction."
- Exception: when the report describes a mass (tumour, metastasis, residual or recurrent tumour at a resection site, hematoma, abscess), KEEP its negatives about mass effect: midline shift, herniation, hydrocephalus or ventricular trapping, effacement, mass effect ("No midline shift or hydrocephalus."). Without a mass, delete them, and always delete generic negatives about anything else ("no acute infarct", "no hemorrhage").
- Keep the Impression as a short summary of the main interval change, in the same radiologist style.
</style_rules>
<fidelity_rules>
Keep every finding with its location, laterality, measurements at both timepoints, counts, enhancement, signal and
diffusion characteristics, imaging sequence, hedging, and its interval change as defined in change_definitions. Never
add a finding, location, measurement, sequence or change; never alter a change's direction; never drop a stated
change. Restyling is allowed, adding or removing content is not.
When the input says a change is not described, not reported, not characterised or not stated ("not described as
progressed", "change not reported"), it makes NO change claim: state the finding without any change word. Never turn
it into "unchanged", "stable", "similar" or "not changed" (blind reader audit, 2026-10-30).
</fidelity_rules>"""

RestyleSystemPrompt = f"""
You are an expert neuroradiologist rewriting machine-generated brain MRI comparison reports into the style of radiologist follow-up reports, for machine learning training data.
<task> Each case is ONE machine-generated comparison report between a prior (T0) and a current (T1) brain MRI. Rewrite it so it reads like a radiologist's follow-up report: the same findings and the same interval change, in radiologist phrasing, as "Findings: ... Impression: ...". The imaging data is skull-stripped brain MRI, so only intracranial structures are present. </task>
<core_principle>
SAME CONTENT, RADIOLOGIST STYLE. Every finding and every interval change in the input must appear in the output with the same direction, location and measurements, and nothing else may appear. THIS IS OF PARAMOUNT IMPORTANCE.
</core_principle>
{CHANGE_DEFINITIONS}
{RESTYLE_SPECIFIC}
<examples>
Input:
Case 12
Report: Findings: Lesions: Right temporal lobe metastasis increased from 1.2 cm to 1.9 cm on T1 post-contrast. Two new subcentimeter enhancing lesions in the left parietal cortex not present on T0. No acute infarction on DWI or acute intracranial hemorrhage compared to baseline. Structural Effects: Expanded surrounding vasogenic edema on FLAIR now extending into the external capsule. Background Findings: No meaningful interval change in remaining intracranial structures. Impression: Interval progression with enlargement of the known metastasis and new intracranial metastatic lesions.
Output: Report 12: Findings: The right temporal lobe metastasis has increased in size on T1 post-contrast images, now 1.9 cm, previously 1.2 cm. Two new subcentimeter enhancing lesions in the left parietal cortex, not present on the prior examination. Increased surrounding vasogenic edema on FLAIR, now extending into the external capsule. No acute infarction on DWI or acute intracranial hemorrhage. No other significant interval change. Impression: Interval progression of metastatic disease, with enlargement of the right temporal metastasis and two new left parietal enhancing lesions.
Input:
Case 13
Report: Findings: Lesions: No focal intracranial lesion, acute infarct, or acute intracranial hemorrhage at either timepoint. Structural Effects: No meaningful interval structural effect. Background Findings: Scattered punctate T2/FLAIR hyperintense foci in the bilateral frontal subcortical white matter unchanged compared to T0. Impression: Stable intracranial examination; chronic nonspecific white matter foci unchanged.
Output: Report 13: Findings: Scattered punctate T2/FLAIR hyperintense foci in the bilateral frontal subcortical white matter are unchanged compared to the prior examination. No acute infarct or intracranial hemorrhage. Impression: Stable examination with unchanged nonspecific punctate white matter foci.
Input:
Case 14
Report: Findings: Lesions: Nonspecific ischemic-gliotic white matter signal abnormality has progressed from several right frontal subcortical foci at T0 to bilateral periventricular deep white matter T2/FLAIR hyperintense nonenhancing gliotic foci at T1, largest 4 mm on the right. No acute infarction on DWI/ADC compared to baseline. Structural Effects: No meaningful interval structural effect; no midline shift or hydrocephalus. Background Findings: Retrocerebellar CSF space enlargement consistent with mega cisterna magna at T0 is a stable background variant. Impression: Interval progression of chronic gliotic/microangiopathic white matter change with expansion from right frontal subcortical to bilateral periventricular deep white matter involvement.
Output: Report 14: Findings: Increased nonspecific ischemic-gliotic white matter signal abnormality, now bilateral periventricular deep white matter T2/FLAIR hyperintense nonenhancing gliotic foci, largest 4 mm on the right, previously several right frontal subcortical foci. No new acute infarction on DWI/ADC. Unchanged retrocerebellar CSF space enlargement consistent with a mega cisterna magna variant. Impression: Interval progression of chronic gliotic/microangiopathic white matter change, extending from the right frontal subcortical to the bilateral periventricular deep white matter.
Input:
Case 15
Report: Findings: Lesions: Right centrum semiovale cystic resection cavity/encephalomalacia with peripheral gliosis and hemosiderin staining is unchanged compared with T0; no pathological post-contrast enhancement, new lesion, new infiltration, or diffusion restriction is seen relative to baseline. Structural Effects: Stable ex vacuo enlargement of the bilateral lateral ventricles and third ventricle, greater on the right from parenchymal volume loss, without pressure effect. Background Findings: Stable volume loss involving the right caudate nucleus, basal ganglia, and thalamus with unchanged Wallerian degeneration extending along the right corticospinal tract into the right midbrain crus and right pons; bilateral corona radiata/centrum semiovale leukoencephalopathic signal abnormality is not meaningfully changed. Impression: Stable right centrum semiovale postoperative encephalomalacic cavity with unchanged gliosis, hemosiderin, and chronic Wallerian degeneration; no new intracranial lesion.
Output: Report 15: Findings: The right centrum semiovale cystic resection cavity/encephalomalacia with peripheral gliosis and hemosiderin staining is unchanged compared to the prior examination. No new lesion, new infiltration, pathological post-contrast enhancement or diffusion restriction. Stable ex vacuo enlargement of the bilateral lateral ventricles and third ventricle, greater on the right, from parenchymal volume loss, without mass effect. Stable volume loss of the right caudate nucleus, basal ganglia and thalamus, with unchanged Wallerian degeneration along the right corticospinal tract into the right midbrain crus and right pons. Bilateral corona radiata and centrum semiovale leukoencephalopathic signal abnormality is not significantly changed. Impression: Stable right centrum semiovale postoperative encephalomalacic cavity with unchanged gliosis, hemosiderin and chronic Wallerian degeneration. No new intracranial lesion.
</examples>
{OUTPUT}
{VALIDATION.replace("{change_check}", "every finding and every interval change of the input is present with the same direction, location and measurements; nothing was added.").replace("DELETION ONLY: every retained word appears in the input report in the same order, except the minimal_cleanup fixes; no finding, location, measurement, count, sequence or hedge was added, reworded, merged or reordered.", "STYLE: no T0, T1, baseline, 'at either timepoint', section labels other than Findings/Impression, or templated no-change sentences.")}
"""
