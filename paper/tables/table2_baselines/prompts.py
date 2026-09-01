"""BrainDiff-equivalent prompt for the proprietary braindiff.models. Carries the SAME content as our
S4 prior-report prompt (models/prompts.py + prior_report_prompts.py): the neuro-radiologist
system prompt, the prior report text, the interval-change guidance, and the Findings/Impression
structure. '/no_think' (Qwen-specific) is dropped."""

SYSTEM = ("You are an expert neuro-radiologist AI assistant. Analyze the provided "
          "neuroimaging study and answer the user's request.")

_STRUCTURE = (
    "Findings: Lesions: [primary lesions/acute infarct/hemorrhage] "
    "Structural Effects: [secondary mass effect/edema/hydrocephalus/etc.] "
    "Background Findings: [chronic contextual findings] "
    "Impression: [clinical interpretation]."
    "Here are two examples out of the output format for your reference:"
    "Findings: Lesions: Numerous chronic nonenhancing demyelinating plaques … unchanged compared with T0; no new intracranial lesion and no enhancing active intracranial plaque … Impression: Stable intracranial demyelinating disease burden without new or active enhancing brain lesions."
    "Findings: Lesions: Small chronic gliotic foci in the frontal subcortical white matter, measuring up to 5 mm on T0, are unchanged in overall burden at T1 where several nonspecific subcortical white matter gliotic foci are again seen bilaterally; no abnormality on SWI at either timepoint and no acute infarction on DWI/ADC or acute intracranial hemorrhage compared to baseline. Structural Effects: No meaningful interval structural effect. Background Findings: Ventricular caliber and age-appropriate sulcal prominence are unchanged. Impression: Stable chronic nonspecific subcortical gliotic white matter foci without acute intracranial abnormality."
    )

_GUIDANCE = ("Infer the current findings from the images and the interval change, and use the "
             "prior report only as the baseline to compare against.")

INSTRUCTION = ("Describe the interval changes observed between the two timepoints, using the "
               "following structure: " + _STRUCTURE)

def user_text(prior_report: str, n_current: int, n_prior: int) -> str:
    """Text block that precedes/labels the images. Mirrors S4: prior report BEFORE the
    instruction, after the images are introduced."""
    pr = prior_report.strip() or "not available."
    imgdesc = (f"You are shown {n_current} axial slices from the CURRENT study"
               + (f" and {n_prior} from the PRIOR study" if n_prior else "")
               + ", labelled by timepoint, modality and slice index.")
    return (f"{imgdesc}\n\nPrior report: {pr}\n\n{_GUIDANCE}\n\n{INSTRUCTION}")
