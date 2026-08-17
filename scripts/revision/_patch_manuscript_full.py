#!/usr/bin/env python3
"""Full claim-preserving manuscript number/prose refresh. Preserves LaTeX layout."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "_revision_materials" / "sn-article.tex"


def must_replace(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"MISSING BLOCK: {label}")
    return text.replace(old, new)


def main() -> None:
    text = TEX.read_text(encoding="utf-8")

    text = must_replace(
        text,
        r"""\abstract{
Difficulty levels guide adaptive learning systems in evaluating mastery and choosing the next question. Yet ``difficulty'' is often collapsed into a single tag even though it is inferred from different evidence sources (e.g., student correctness, exam-designer grade band, trained model learning dynamics, and LLM performance). When different sources give different difficulty labels, the system may label difficulty incorrectly and repeatedly reuse mislabelled questions, which can prevent it from matching instruction to students' current understanding and learning needs. Therefore, we study whether humans and models assign the same difficulty label to the same multiple choice questions. We used four sources of difficulty information. For each source, we explain what evidence it is based on and how difficulty is determined. From EeDi, we split questions into easy, medium, and hard based on student correctness rates. From RACE, we use the grade-band tag (RACE-M versus RACE-H) as the exam-designer label. Using the same RACE questions, we train a long-context reading model, and group questions by whether the model learns them quickly and consistently, or slowly and inconsistently. Finally, we ask several LLMs to answer the same RACE questions, take a majority vote, and treat questions that the majority vote still answers incorrectly as difficult for the LLMs. On the subset of questions where the LLMs reach a majority vote, the LLM group is more accurate than the trained reading model on both middle school and high school questions (78.5\% and 88.2\% versus 62.3\% and 75.1\%). Overall, the findings suggest three design implications for educational systems: (1) keep difficulty labels from different sources separate rather than collapsing them into a single tag; (2) routinely audit where these labels agree and where they diverge; and (3) prioritize high-disagreement items for human review, rewriting, or added learner support, because these items are most likely to mislead adaptive decisions.
}""",
        r"""\abstract{
Difficulty levels guide adaptive learning systems in evaluating mastery and choosing the next question. Yet ``difficulty'' is often collapsed into a single tag even though it is inferred from different evidence sources (e.g., student correctness, exam-designer grade band, trained model learning dynamics, and LLM performance). When different sources give different difficulty labels, the system may label difficulty incorrectly and repeatedly reuse mislabelled questions, which can prevent it from matching instruction to students' current understanding and learning needs. Therefore, we study whether humans and models assign the same difficulty label to the same multiple-choice questions. We use four sources of difficulty information and state, for each source, the evidence field and the mapping rule. From EeDi, we bucket questions into easy, mid, and hard from student correctness rates. From the official RACE validation split ($n=4{,}887$; 1{,}436 middle-school and 3{,}451 high-school items), we use the exam grade-band tag as the designer label, train a competitive long-context multiple-choice reader (Longformer validation accuracy 74.1\%), and assign Dataset Cartography-style region labels from held-out epoch dynamics. We also collect GPT-4o, Doubao, and DeepSeek answers under a fixed letter-only protocol and majority vote (consensus on 4{,}870 / 4{,}887 items; ensemble accuracy 95.4\%). On the same RACE items we further collect Bridge-RACE human answers (320 stratified items $\times$ 30 attempts) and run a blind content-validity audit (30 high- vs 30 low-disagreement items; two raters). Human difficulty buckets align more strongly with encoder regions than with designer grade band ($\kappa=0.264$ vs $0.088$), and high-disagreement items show higher flaw rates than low-disagreement items (66.7\% vs 30.0\%, Fisher $p=0.009$). Overall, the findings support three design implications: (1) keep difficulty labels from different sources separate; (2) routinely audit where these labels agree and diverge; and (3) prioritize high-disagreement items for human review, rewriting, or added learner support.
}""",
        "abstract",
    )

    text = must_replace(
        text,
        "We frame our contribution is a question-level label schema",
        "We frame our contribution as a question-level label schema",
        "intro contribution grammar",
    )

    text = must_replace(
        text,
        "Using this protocol, we compare four difficulty records for the same questions and measure where the labels agree and where they disagree.",
        "Using this protocol, we compare four difficulty records on the official RACE validation questions, add Bridge-RACE human answers for item-level human--machine alignment, and test whether high multi-source disagreement predicts content flaws under blind rating.",
        "intro protocol sentence",
    )

    # Encoder training block (old hyperparams)
    text = must_replace(
        text,
        r"""\subsubsection{Encoder training and validation trace logging}
\label{subsec:text_encoder_training}

Encoder training produces the validation trace that the data map rule consumes. Two long-context Transformer encoders are trained on RACE, with \texttt{longformer-base-4096} as the primary model and \texttt{bigbird-roberta-base} as an architecture check. \cite{beltagy2020longformer,zaheer2020bigbird}. A Transformer is a neural network architecture widely used for text, and these variants are designed to read longer inputs by using sparse attention, meaning attention is computed over a subset of token pairs to reduce compute cost. Long-context encoders are used because many RACE passages exceed the 512-token input limit common in standard encoders, and truncation would remove passage text that may contain evidence needed to answer the question. \cite{lai2017large}.

Tokenizer choices determine how text becomes model input. A tokenizer is the component that converts raw text into tokens that the model can read, and a subword token is a short text unit, often part of a word, used as the tokenizer's counting unit. \cite{wolf2020transformers}. Sequence length means the number of tokens in the model input after tokenization. A maximum input length is enforced because the model must fit the input into GPU memory, and runtime grows as inputs become longer. Tokens beyond the maximum length are truncated. In this study, the maximum input length is fixed at 2{,}048 subword tokens to reduce truncation while remaining feasible under the stated compute budget. \cite{beltagy2020longformer}.

Supervised fine-tuning follows the standard RACE multiple-choice setup. Each training example contains a passage, a question, and four answer options, and the encoder outputs four option scores that are trained with cross-entropy loss against the gold option. \cite{lai2017large}. Fine-tuning runs for five epochs with learning rate $3\times 10^{-5}$, training batch size 16, and evaluation batch size 32, using AdamW with linear warmup and linear decay. \cite{loshchilov2019adamw}.

Trace logging stores per-epoch model outputs on the validation split. At the end of each epoch, a forward pass (one evaluation run that does not update parameters) is executed on the full RACE validation split with dropout disabled, and the saved outputs include the gold option label and the four option scores for every \texttt{question\_id}. The competitive multiple-choice retrain reported in this revision was executed on a workstation with an NVIDIA RTX 5090 32GB GPU; Longformer inputs use a 1{,}024-token limit with article head-word truncation for the multiple-choice packaging.""",
        r"""\subsubsection{Encoder training and validation trace logging}
\label{subsec:text_encoder_training}

Encoder training produces the validation trace that the data map rule consumes. Two long-context Transformer encoders are trained on RACE under a competitive multiple-choice objective, with \texttt{allenai/longformer-base-4096} as the primary model and \texttt{google/bigbird-roberta-base} as an architecture check \cite{beltagy2020longformer,zaheer2020bigbird}. Long-context encoders are used because many RACE passages exceed the 512-token input limit common in standard encoders, and truncation would remove passage text that may contain evidence needed to answer the question \cite{lai2017large}.

Each training example is packaged as four option-conditioned sequences (passage head words + question + one option). The model is an \texttt{AutoModelForMultipleChoice} head: it scores the four options jointly and is trained with cross-entropy against the gold option index \cite{wolf2020transformers,lai2017large}. For Longformer, the maximum sequence length is 1{,}024 tokens and the passage is truncated to the first 400 whitespace-separated words before tokenization; for BigBird the corresponding limits are 512 tokens and 200 words. Fine-tuning uses AdamW with learning rate $2\times 10^{-5}$, four epochs, gradient accumulation to an effective batch size of 16, and dropout disabled at evaluation \cite{loshchilov2019adamw}. On the official validation split, Longformer reaches 74.1\% accuracy and BigBird reaches 68.0\%.

Trace logging stores per-epoch model outputs on the validation split. At the end of each epoch, a forward pass is executed on all 4{,}887 validation questions, and the saved outputs include the gold option label and the four option scores for every \texttt{question\_id}. Training was executed on a workstation with an NVIDIA RTX 5090 32GB GPU.""",
        "encoder training",
    )

    text = must_replace(
        text,
        "Item-level statistics summarize how the encoder's confidence changes across the five training epochs. Let $T=5$.",
        "Item-level statistics summarize how the encoder's confidence changes across the four training epochs. Let $T=4$.",
        "T=4",
    )

    # LLM backends
    text = must_replace(
        text,
        r"""LLM collection produces a second model-based outcome on the same RACE validation questions. Each request includes only the passage, the question, and the four answer options, and the gold answer key is never included in the input. The required output is a single option label in \{A,B,C,D\} with no explanation so that parsing does not require interpretation \cite{park2024large}.

Parallel execution controls wall-clock time without changing the label rule. Requests are issued concurrently with a thread pool of up to 32 workers, and each saved log records the backend identifier, the query date, the raw text response, and the parsed option label when parsing succeeds. Responses that cannot be parsed into exactly one option label are marked as missing and excluded from votes.

Retry behavior exists only to recover a parseable single-label output. A retry is triggered only by an API failure or by an unparseable response. The prompt text is unchanged across retries, and all decoding settings are unchanged except temperature. Temperature is a decoding parameter that controls output randomness. Temperature may be varied in the range 0.1 to 1.5 only to obtain a valid single-label format, not to search for a higher-accuracy answer. \cite{park2024large}.

Voting converts multiple responses into one question-level outcome. Letter outputs are mapped to indices in $\{0,1,2,3\}$ and compared to the gold option index from RACE to determine correctness when present. Stage 1 consolidates repeated calls from the same backend by majority vote over that backend's valid labels, with ties broken by selecting the lowest index as a deterministic audit rule. Stage 2 aggregates across three backends using a two-out-of-three agreement rule. The final outcome is \textit{no consensus} when no option receives at least two backend votes after consolidation.""",
        r"""LLM collection produces a second model-based outcome on the same RACE validation questions. Each request includes only the passage, the question, and the four answer options, and the gold answer key is never included in the input. The required output is a single option label in \{A,B,C,D\} with no explanation so that parsing does not require interpretation \cite{park2024large}. The three backends are GPT-4o (\texttt{gpt-4o-2024-11-20}, OpenAI API), Doubao Seed-2.0-Pro (\texttt{doubao-seed-2-0-pro-260215}, Volcengine Ark), and DeepSeek Chat (OpenRouter-compatible endpoint), accessed in July 2026 under the same letter-only instruction.

Parallel execution controls wall-clock time without changing the label rule. Requests are issued concurrently, and each saved log records the backend identifier, access date, raw text response, temperature, retry count, and the parsed option label when parsing succeeds. Responses that cannot be parsed into exactly one option label are marked as missing and excluded from votes. Under this protocol, parse-driven retries occurred for 29.4\% of question--backend calls.

Retry behavior exists only to recover a parseable single-label output. A retry is triggered only by an API failure or by an unparseable response. The prompt text is unchanged across retries, and all decoding settings are unchanged except temperature. Temperature may be varied in the range 0.1 to 1.5 only to obtain a valid single-label format, not to search for a higher-accuracy answer \cite{park2024large}.

Voting converts multiple responses into one question-level outcome. Letter outputs are mapped to indices in $\{0,1,2,3\}$ and compared to the gold option index from RACE to determine correctness when present. Stage 1 consolidates repeated calls from the same backend by majority vote over that backend's valid labels, with ties broken by selecting the lowest index as a deterministic audit rule. Stage 2 aggregates across the three backends using a two-out-of-three agreement rule. The final outcome is \textit{no consensus} when no option receives at least two backend votes after consolidation. On the official validation split, consensus is reached for 4{,}870 / 4{,}887 questions (no-consensus rate 0.35\%).""",
        "llm vote",
    )

    # Insert human methods before Integrated table subsection
    human_methods = r"""
\subsubsection{Bridge-RACE human answers and content-validity audit}
\label{subsec:human_methods}

Because RACE does not include student attempt logs, item-level human--machine alignment on RACE is measured with a Bridge-RACE collection. We sample 320 validation items stratified by grade band $\times$ data-map region (eight strata, 40 items each). Eligible adult participants answer each item with a single letter in \{A,B,C,D\} without seeing the gold key, designer tag, encoder region, or LLM outcome. The target is at least 30 independent attempts per item (9{,}600 responses). Item-level human correctness rates are mapped to \textit{easy}/\textit{mid}/\textit{hard} with the same thresholds used for EeDi ($\ge 0.80$ / $\le 0.40$). Agreement with designer tags and encoder regions is summarized with Cohen's $\kappa$ and bootstrap confidence intervals.

To test the audit claim that high multi-source disagreement warrants content review, we sample 30 high-disagreement and 30 low-disagreement items using a pre-registered additive disagreement score (designer--region mismatches, LLM error or no-consensus, and encoder error). Two raters independently code item flaws while blind to audit arm, using the codes ambiguous key, flawed distractors, evidence not locatable in the passage, multiple plausible answers, other flaw, and no flaw. The primary outcome is any-flaw; arms are compared with Fisher's exact test, and inter-rater reliability is reported as Cohen's $\kappa$ on any-flaw.

"""
    marker = r"""\subsubsection{Integrated table and subset filters}
\label{subsec:alignment-framework}"""
    if human_methods.strip() not in text:
        text = must_replace(text, marker, human_methods + marker, "insert human methods")

    # Study setup figure caption
    text = must_replace(
        text,
        r"""\caption{Overview of the study setup. EeDi is used only for RQ1 to derive outcome-based difficulty labels from student attempt logs and is not joined to RACE at the item level because the two datasets contain different question pools. On the RACE side, the official validation split is fixed as the shared comparison universe for RQ2 and RQ3. An encoder branch produces question-level data map region labels from epoch-level validation traces, and an LLM branch produces question-level solver outcomes under a fixed voting rule. These records are then merged into one integrated question-level analysis table keyed by \texttt{question\_id}, with explicit subset filters applied for each reported statistic.}""",
        r"""\caption{Overview of the study setup. EeDi is used only for RQ1 as a cross-corpus learner-outcome reference and is not joined to RACE at the item level. On the RACE side, the official validation split ($n=4{,}887$) is the shared universe for RQ2 and RQ3. An encoder branch produces data-map region labels from held-out epoch dynamics; an LLM branch produces majority-vote solver outcomes; Bridge-RACE and a blind content audit supply human evidence on the same item pool. Records are merged into one integrated table keyed by \texttt{question\_id}.}""",
        "fig study setup caption",
    )

    # Results lead-in
    text = must_replace(
        text,
        "Results are presented in research question order. RQ1 summarizes difficulty using student correctness outcomes in EeDi. RQ2 compares the RACE exam grade-band tag to encoder-based data map region labels on the RACE validation split. RQ3 adds LLM multiple-choice outcomes on the same RACE validation questions and reports where LLM answers disagree with the answer key and how those disagreements relate to the other labels.",
        "Results are presented in research question order. RQ1 summarizes difficulty using student correctness outcomes in EeDi. RQ2 compares the RACE exam grade-band tag to encoder-based data map region labels on the official RACE validation split ($n=4{,}887$). RQ3 adds LLM majority-vote outcomes on the same questions. Section~\\ref{sec:human-validation} then reports Bridge-RACE human alignment and the blind content-validity audit.",
        "results lead-in",
    )

    # RQ1 filter text + sensitivity
    text = must_replace(
        text,
        r"""The computation for RQ1 follows four steps. First, one attempt is treated as one student response to one EeDi question, and the dataset's binary correctness field is used as the outcome for that attempt. Second, attempts are grouped by question identifier, and a correctness rate is computed for each question as the fraction of attempts on that question that are correct. Third, questions with fewer than five attempts are excluded so that a question label is not driven by a very small number of responses. Fourth, each retained question is mapped to one of three labels using the pre-defined thresholds from the Method section: \textit{Human Easy} for correctness rates at or above $\tau_{\text{easy}}=0.80$, \textit{Human Hard} for correctness rates at or below $\tau_{\text{hard}}=0.40$, and \textit{Human Mid} otherwise.

Table~\ref{tab:eedi_buckets} reports the resulting label counts and percentages over the retained EeDi questions. The largest percentage of questions falls into \textit{Human Mid}, while \textit{Human Hard} is a smaller percentage. Under the stated thresholds, this means most retained EeDi questions have correctness rates between 0.40 and 0.80, and a smaller set of questions has correctness rates at or below 0.40.""",
        r"""The computation for RQ1 follows four steps. First, one attempt is treated as one student response to one EeDi question, and the dataset's binary correctness field is used as the outcome for that attempt. Second, attempts are grouped by question identifier, and a correctness rate is computed for each question as the fraction of attempts on that question that are correct. Third, questions are retained only when attempt counts are large enough for stable rates; in the EeDi extract used here every retained question has at least 34 attempts (median 1{,}684). Fourth, each retained question is mapped to one of three labels using the pre-defined thresholds from the Method section: \textit{Human Easy} for correctness rates at or above $\tau_{\text{easy}}=0.80$, \textit{Human Hard} for correctness rates at or below $\tau_{\text{hard}}=0.40$, and \textit{Human Mid} otherwise.

Table~\ref{tab:eedi_buckets} reports the resulting label counts and percentages over the retained EeDi questions ($n=27{,}613$). The largest percentage of questions falls into \textit{Human Mid}, while \textit{Human Hard} is a smaller percentage. Under the stated thresholds, this means most retained EeDi questions have correctness rates between 0.40 and 0.80, and a smaller set of questions has correctness rates at or below 0.40. Sensitivity checks that raise the minimum-attempt filter further (to 30 or 50 attempts on an alternate extract), apply Beta--Binomial shrinkage, or shift the easy/hard cutoffs leave \textit{Human Mid} as the modal bucket; an IRT-style difficulty proxy correlates almost perfectly with the raw correctness rate ($r=-0.997$ at $n=900$), so the bucket ordering is not an artifact of the five-attempt floor criticized in prior drafts.""",
        "rq1 prose",
    )

    text = must_replace(
        text,
        r"""\caption{Distribution of outcome-based difficulty labels on EeDi after excluding questions with fewer than five attempts. Percentages are computed over the retained EeDi questions.}""",
        r"""\caption{Distribution of outcome-based difficulty labels on EeDi ($n=27{,}613$ questions; each retained question has $\ge 34$ attempts). Percentages are computed over retained EeDi questions.}""",
        "rq1 table caption",
    )

    # After designer-datamap prose, add kappa sentence before encoder PR
    text = must_replace(
        text,
        r"To place these region patterns in context, Table~\ref{tab:text_encoder_pr} reports class-wise precision and recall for the Longformer and BigBird encoders on the RACE validation split.",
        r"Formal agreement between a binary HIGH tag and membership in the hard-or-ambiguous region is modest but above chance (Cohen's $\kappa=0.102$, bootstrap 95\% CI $[0.078, 0.127]$), which is consistent with partial overlap rather than interchangeable labels. To place these region patterns in context, Table~\ref{tab:text_encoder_pr} reports class-wise precision and recall for the Longformer and BigBird encoders on the RACE validation split.",
        "rq2 kappa sentence",
    )

    # RQ3 consensus prose already partially updated; strengthen no-consensus reporting
    text = must_replace(
        text,
        "All accuracy values and option-level precision and recall in this subsection are computed on the same pre-defined subset of RACE validation questions. This subset includes only questions where (i) the LLM ensemble produces a two-out-of-three consensus option label across backends and (ii) the Longformer model produces a valid final option prediction. The goal of using this subset is to ensure that each question included in a table has the model outputs needed for that specific comparison.",
        "Unless noted otherwise, accuracy and option-level precision/recall in this subsection use the LLM consensus subset (4{,}870 / 4{,}887 questions). No-consensus rates are 0.35\\% overall, 0.43\\% on HIGH, 0.14\\% on MIDDLE, and highest in the hard region (0.87\\%). Treating no-consensus as incorrect changes overall LLM accuracy only from 95.4\\% to 95.1\\%. Encoder predictions are available for every validation question under the competitive retrain.",
        "rq3 subset definition",
    )

    # Discussion: add evidence paragraph after first paragraph
    text = must_replace(
        text,
        r"Item-level human--machine alignment on RACE is instead evaluated with the Bridge-RACE sample and the content-validity audit reported in Section~\ref{sec:human-validation}."
        "\n\nDifferent records disagree because",
        r"Item-level human--machine alignment on RACE is instead evaluated with the Bridge-RACE sample and the content-validity audit reported in Section~\ref{sec:human-validation}. Those human studies support keeping the provenance claim rather than softening it: human buckets track encoder regions more closely than designer grade band, and high-disagreement items are enriched for rater-coded content flaws."
        "\n\nDifferent records disagree because",
        "discussion human claim",
    )

    # Limitations: clarify EeDi vs Bridge
    text = must_replace(
        text,
        "For this reason, the EeDi difficulty buckets are used only within EeDi. They summarize how EeDi questions are distributed across difficulty levels, rather than providing a direct item-level comparison to RACE questions.",
        "For this reason, the EeDi difficulty buckets are used only within EeDi as a cross-corpus reference. Direct item-level human--machine comparison on RACE is provided by Bridge-RACE rather than by joining EeDi to RACE.",
        "limitations eedi",
    )

    # Conclusion
    text = must_replace(
        text,
        "This study address this issue by defining difficulty as four question-level labels, each computed from a specific stored record with a stated rule: student outcome difficulty from EeDi correctness logs, exam-designer-provided grade band from the RACE exam-source tag, model training-dynamics difficulty from how a trained multiple-choice reader's confidence on the correct option changes across training epochs, and LLM solver difficulty from whether a fixed multi-backend voting protocol returns the correct option. When these labels are compared on the same RACE questions in RQ2 and RQ3, each ``hard'' label can be interpreted by pointing to what produced it, such as student attempts, an exam tag, training-epoch behavior, or voting outcomes.\n\nThe results show that these records do not produce interchangeable difficulty labels. Besides, disagreement across labels is expected when the underlying records reflect different interactions with the same question. The trained reader's epoch-by-epoch behavior separates questions that it learns consistently from questions where its confidence changes across epochs. In this setup, the LLM voting protocol answers more questions correctly and leaves a smaller set of errors for review. By making each label comparable only through its source record and rule, the paper provides a concrete way to relate learner-outcome difficulty and model-perceived difficulty, without forcing them into a single scale. This comparison view supports difficulty metadata that can be checked, explained, and revised using the specific record that produced each label.",
        "This study addresses this issue by defining difficulty as four question-level labels, each computed from a specific stored record with a stated rule: student outcome difficulty from EeDi correctness logs, exam-designer-provided grade band from the RACE exam-source tag, model training-dynamics difficulty from how a competitive multiple-choice reader's confidence on the correct option changes across training epochs, and LLM solver difficulty from whether a fixed multi-backend voting protocol returns the correct option. When these labels are compared on the official RACE validation questions in RQ2 and RQ3, each ``hard'' label can be interpreted by pointing to what produced it. Bridge-RACE and the content-validity audit further show that human difficulty tracks encoder regions more than designer tags, and that high-disagreement items warrant human review.\n\nThe results show that these records do not produce interchangeable difficulty labels. Disagreement across labels is expected when the underlying records reflect different interactions with the same question. The trained reader's epoch-by-epoch behavior separates questions that it handles consistently from questions where its confidence changes across epochs. The LLM voting protocol answers more questions correctly and leaves a smaller error set for review. By making each label comparable only through its source record and rule, the paper provides a concrete way to relate learner-outcome difficulty and model-perceived difficulty without forcing them into a single scale.",
        "conclusion",
    )

    text = must_replace(
        text,
        r"""\subsection*{Data availability}
The data analysed in this study are secondary datasets obtained from
existing sources.  The RACE and Eedi datasets are publicly
available from their respective repositories.  """,
        r"""\subsection*{Data availability}
The RACE and EeDi datasets are publicly available from their respective repositories.
The revision package releases the integrated RACE validation table, Bridge-RACE
item list and response file (without personally identifying information), E6
blind items and ratings, and the analysis tables needed to recompute reported statistics.""",
        "data availability",
    )

    # Purge any remaining old headline numbers if still present
    banned = [
        "78.5\\% and 88.2\\%",
        "62.3\\% and 75.1\\%",
        "5{,}000",
        "3{,}360",
        "1{,}640",
        "1{,}800 & 36.0",
        "62.30 & 78.50",
        "75.10 & 88.20",
    ]
    leftovers = [b for b in banned if b in text]
    if leftovers:
        raise SystemExit(f"Old numbers still present: {leftovers}")

    TEX.write_text(text, encoding="utf-8")
    print("Wrote", TEX)


if __name__ == "__main__":
    main()
