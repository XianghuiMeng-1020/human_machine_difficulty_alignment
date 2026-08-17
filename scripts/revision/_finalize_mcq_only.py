#!/usr/bin/env python3
"""Finalize manuscript: English polish + MCQ-only scope; no open-ended work."""
from __future__ import annotations

from pathlib import Path

TEX = Path(__file__).resolve().parents[2] / "_revision_materials" / "sn-article.tex"


def repl(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        print(f"SKIP {label}")
        return text
    print(f"OK {label}")
    return text.replace(old, new)


def main() -> None:
    text = TEX.read_text(encoding="utf-8")
    # normalize NBSP only; do not collapse spaces (preserves LaTeX table alignment)
    text = text.replace("\u00a0", " ")

    fixes = [
        (
            "and the record indicates whether the answer matches the answer key or misses it.\\cite{strugatski2024turing}.",
            "and the record indicates whether the answer matches the answer key or misses it~\\cite{strugatski2024turing}.",
            "cite space intro1",
        ),
        (
            "They are not always checked for agreement across different difficulty label producers for the same question. \\cite{benedetto2023survey,alkhuzaey2024text}.",
            "They are not always checked for agreement across different difficulty label producers for the same question~\\cite{benedetto2023survey,alkhuzaey2024text}.",
            "cite space intro2",
        ),
        (
            "and can make it harder to check each question carefully before reuse. \\cite{indran2024twelvetips,lee2024edm_mcq,bitew2023distractor}.",
            "and can make it harder to check each question carefully before reuse~\\cite{indran2024twelvetips,lee2024edm_mcq,bitew2023distractor}.",
            "cite space intro3",
        ),
        (
            "comparing four records  of describing question difficulty",
            "comparing four records that describe question difficulty",
            "four records phrasing",
        ),
        (
            "The second record is a exam-designer label record",
            "The second record is an exam-designer label record",
            "a exam -> an exam",
        ),
        (
            "Learning-sciences research often argues that: how difficult a question, is depends on who is answering it and in what context, rather than being determined solely by the wording of the question",
            "Learning-sciences research often argues that how difficult a question is depends on who is answering it and in what context, rather than being determined solely by the wording of the question",
            "question is depends",
        ),
        (
            "work on cognitive load mention that difficulty can reflect",
            "work on cognitive load notes that difficulty can reflect",
            "mention -> notes",
        ),
        (
            "Exam-designer-provided grade band is taken directly from RACE as a two-level label that indicates the exam source grade band for each question. The two values are \\textit{MIDDLE} (middle school exam source) and \\textit{HIGH} (high school exam source). \\cite{lai2017large}.",
            "Exam-designer-provided grade band is taken directly from RACE as a two-level label that indicates the exam source grade band for each question. The two values are \\textit{MIDDLE} (middle school exam source) and \\textit{HIGH} (high school exam source)~\\cite{lai2017large}.",
            "cite designer",
        ),
        (
            "It reduces sudden jumps from ``too easy'' to ``too hard'' caused by wrong difficulty tags, so practice feels more smooth and consistent across weeks and cohorts.",
            "It reduces sudden jumps from ``too easy'' to ``too hard'' caused by wrong difficulty tags, so practice feels smoother and more consistent across weeks and cohorts.",
            "more smooth",
        ),
        (
            "can combine two evidence:",
            "can combine two kinds of evidence:",
            "two evidence",
        ),
        (
            "For constructed-response question,",
            "For a constructed-response question,",
            "constructed-response question",
        ),
    ]
    for old, new, label in fixes:
        text = repl(text, old, new, label)

    # Explicit MCQ scope in Method datasets
    text = repl(
        text,
        r"""Two open-source multiple-choice datasets are used in this study, because the analysis requires two different question-level difficulty labels. Here, open-source means that the dataset is publicly released under a license that permits reuse \cite{pmlr-v133-wang21a,lai2017large}. EeDi provides student attempt logs, which allow difficulty to be defined from learner outcomes for RQ1 (how often students answer each question correctly). RACE provides full question text and an exam source grade-band tag (middle school versus high school), which serves as an exam-designer-provided label for RQ2 and RQ3. A single dataset cannot support all three research questions, because EeDi does not include an exam source grade-band tag and RACE does not include platform-scale student attempt logs.""",
        r"""This study is scoped to multiple-choice questions (MCQs) only. Option-level probabilities, letter-only LLM voting, and Bridge-RACE human answers all require a fixed four-option format; open-ended or constructed-response items are outside the present design. Two open-source MCQ datasets are used, because the analysis requires two different question-level difficulty labels. Here, open-source means that the dataset is publicly released under a license that permits reuse \cite{pmlr-v133-wang21a,lai2017large}. EeDi provides student attempt logs, which allow difficulty to be defined from learner outcomes for RQ1 (how often students answer each question correctly). RACE provides full question text and an exam source grade-band tag (middle school versus high school), which serves as an exam-designer-provided label for RQ2 and RQ3. A single dataset cannot support all three research questions, because EeDi does not include an exam source grade-band tag and RACE does not include platform-scale student attempt logs.""",
        "MCQ scope in datasets",
    )

    # Replace Limitations + Future Work with MCQ-only framing
    old_lim = r"""\section{Limitations and Future Work}
\label{sec:limitations}

This study describes disagreement between difficulty labels that come from different stored records, and the scope is limited to the two datasets and protocols analyzed. We study multiple choice questions in two contexts. The first context is RACE, a set of English reading-comprehension exam questions. Each RACE question includes an exam-source grade-band tag: RACE-M (MIDDLE) or RACE-H (HIGH). The second context is EeDi, a set of K–12 diagnostic practice questions. In EeDi, student attempt logs are available, so we can compute a per-question correctness rate from many student attempts. By focusing on these two datasets, we can define every difficulty label in a concrete and reproducible way. For each label, we can name the exact input field we use and the explicit rule that turns that field into a label. However, this choice also limits what we cover. We do not study response formats that produce different kinds of outputs, such as short-answer explanations, open-response proofs, programming submissions, or multi-step solutions with intermediate work. EeDi and RACE also contain different question sets, so their questions do not match one-to-one. For this reason, the EeDi difficulty buckets are used only within EeDi as a cross-corpus reference. Direct item-level human--machine comparison on RACE is provided by Bridge-RACE rather than by joining EeDi to RACE.

The model-based results in this paper depend on the specific training and labeling choices we made for the RACE-side records. They may change if we use a different model, a different training setup, or a different labeling rule. In particular, the encoder-centered label is based on one family of trained multiple choice reading models and one fixed data map rule. That rule uses the model's probability on the correct option, tracked across training epochs. The LLM-centered outcome is derived from one fixed prompt format, one fixed requirement to return a single option label, and one fixed voting rule. Changes to the reader architecture, the number of training epochs, the cut points used for region assignment, the set of LLM backends, or the decoding settings could cause some questions near cut points or near voting ties to flip labels. Finally, while the content-validity audit shows higher flaw rates among high-disagreement items, and an offline policy simulation compares review prioritization rules, the paper does not yet test whether acting on those flags changes student learning, teacher workload, or live system error rates. Addressing this would require a separate study with an explicit instructional or platform decision rule and an outcome measure.

Therefore, future work can broaden the same comparison logic by adding new formats while keeping each difficulty label tied to a clearly logged record and a clearly stated mapping rule. For constructed-response question, the student-facing record could be a rubric-scored correctness label, recorded over many attempts, and mapped into buckets using pre-specified thresholds. For programming tasks, a student-facing record can combine two kinds of evidence: whether each submission passes the tests and how long the student takes to solve the task. The system logs both evidence for each attempt and then summarizes them for the task overall. For multi-step problems, a student-facing record can track performance at the step level. For example, it can store the sequence of step successes and failures, which shows where students often get stuck, stop, or need to revise their work. In parallel, the LLM-based outcome can be stress-tested by changing one factor at a time, such as prompt wording, the inclusion of a small set of few-shot examples, temperature, or the set of LLM backends, and then reporting how often a question changes its voted label under the same voting rule. Another extension is to train a dedicated difficulty predictor, with targets drawn from the same human-derived records used here (e.g., RACE grade band and EeDi difficulty buckets), and then compare its disagreement patterns to the training-dynamics labels under the same audit definitions. A final extension is to move beyond question-only summaries and model difficulty as something that can vary across learners. This can be done by combining question-level evidences with student history within a single dataset that contains both types of information. A follow-up step is then to define a decision rule in advance that uses these combined signals, and to test in a controlled study whether applying that rule improves downstream outcomes."""

    new_lim = r"""\section{Limitations and Future Work}
\label{sec:limitations}

This study is intentionally limited to multiple-choice questions. All primary analyses---encoder option scoring, letter-only LLM voting, Bridge-RACE human answers, and the content-validity audit---assume a fixed four-option format with a gold key. We do not evaluate open-ended, short-answer, programming, or multi-step constructed-response items, and we do not claim that option-probability cartography transfers to those formats. Within the MCQ scope, we use two datasets. RACE supplies English reading-comprehension exam items with MIDDLE/HIGH grade-band tags for RQ2 and RQ3. EeDi supplies K--12 diagnostic MCQ attempt logs for the RQ1 learner-outcome reference. The two item pools do not match one-to-one, so EeDi is not joined to RACE at the item level; item-level human--machine alignment on RACE is provided by Bridge-RACE instead.

Model-based MCQ results depend on the training and labeling choices we made. The encoder-centered label uses one competitive multiple-choice reader family and one fixed data-map rule on held-out epoch dynamics. The LLM outcome uses one letter-only prompt, three backends, and a fixed two-out-of-three vote. Changing the reader, cut points, backends, or decoding settings can flip labels near boundaries or voting ties. The content-validity audit and the offline review-policy comparison support prioritizing high-disagreement MCQs for human review, but they do not measure whether acting on those flags improves student learning, teacher workload, or live platform error rates in a controlled deployment.

Future work within the MCQ setting can stress-test the LLM vote by varying prompt wording, few-shot exemplars, temperature, or backend sets while keeping the same voting rule; train an MCQ difficulty predictor against Bridge or designer labels and compare disagreement patterns with training-dynamics regions; and run a pre-registered classroom or platform A/B test of disagreement-triggered review on MCQ banks. Those extensions keep the same provenance principle---store each difficulty label with its source and rule---without leaving the multiple-choice scope of this paper."""

    text = repl(text, old_lim, new_lim, "limitations MCQ-only")

    # Intro: one sentence that scope is MCQ
    text = repl(
        text,
        "We frame our contribution as a question-level label schema and cross-source comparison protocol that stores each difficulty label together with (1) its source (students, exam designers, trained readers, or LLM solvers) and (2) the concrete dataset field and rule used to compute it.",
        "We frame our contribution as a question-level label schema and cross-source comparison protocol for MCQs that stores each difficulty label together with (1) its source (students, exam designers, trained readers, or LLM solvers) and (2) the concrete dataset field and rule used to compute it.",
        "contribution MCQ",
    )

    TEX.write_text(text, encoding="utf-8")
    print("Wrote", TEX)


if __name__ == "__main__":
    main()
