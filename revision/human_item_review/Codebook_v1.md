# Blinded Item-Review Codebook

Version 1 — freeze after the pilot discussion and before formal coding.

This is the complete coder-facing codebook for the blinded RACE item-review study associated with:

- Manuscript: *Who Finds It Hard? Mapping Human–Machine Alignment in Question Difficulty*
- Manuscript ID: FODE-D-26-00032

The official coder file is `Codebook_v1.docx`. This Markdown copy is identical in content and is provided so the codebook can be read on GitHub.

The codebook does not ask coders to judge whether an item is globally easy or hard, to validate model labels, or to infer any study grouping.

---

## 1. Purpose of the review

This review asks whether specific content problems are present in a multiple-choice reading item. The review is item-centered. It is not a rating of student ability, not a rating of overall item difficulty, and not a check of any automated score.

## 2. What you will and will not judge

**You will judge:**

- Which option you believe is the best answer from the given passage (Stage 1).
- How confident you are in that answer.
- What kind of evidence the answer requires.
- Whether a rendering or content problem prevents a reasonable review.
- Whether the official key, the question wording, the distractors, or the passage evidence has a defined review concern (Stage 2).

**You will not judge:**

- Whether the question is globally easy or hard.
- Whether a computer or other rater would get the item right.
- An overall quality grade or star rating.
- The source of the item or any grouping used by the research team.

## 3. Two-stage procedure

Stage 1 is completed with the official answer hidden. After you return Stage 1, you receive Stage 2, which shows the official key. Do not change Stage 1 answers after you have seen Stage 2. If you notice a typo in Stage 1 after submission, tell the research team rather than editing quietly.

## 4. Stage 1 codes

### 4.1 rater_answer

Allowed values: A, B, C, D. Choose the single best answer from the four options using only the passage and the question.

Boundary: If two options seem possible, still choose one and lower your confidence. Use `cannot_judge` only when the item cannot be reviewed reasonably.

### 4.2 answer_confidence

- **1 = low confidence.** You are guessing or see no clear basis.
- **2 = somewhat unsure.** You have a lean but can see a serious alternative.
- **3 = fairly confident.** Another option is possible but clearly weaker.
- **4 = very confident.** The chosen option is clearly best from the given text.

### 4.3 evidence_demand

**1 = local explicit evidence**

The needed evidence is directly stated in one local part of the passage. Example: The passage says “The shop closes at 6 p.m.” The question asks when the shop closes. The answer is stated in one sentence.

**2 = local inference**

The answer requires a reasonable inference from a nearby part of the passage. Example: A sentence says Maya left after sunset, and an earlier nearby sentence says sunset was at 7 p.m. The question asks whether she left before 6 p.m. The reader must infer “no” from those nearby details.

**3 = distributed evidence**

The answer requires combining information from more than one part of the passage. Example: Paragraph 1 states a rule, and paragraph 3 states an exception. The question can be answered only by using both parts. Distributed evidence is a description of how the item works. It is not, by itself, a defect.

**4 = outside-passage knowledge**

Important information not provided in the passage is needed. Example: The question asks which city is the capital, and the passage never states or implies it. Ordinary decoding of common words is not outside-passage knowledge.

### 4.4 cannot_judge

- **0** = the item can be reviewed normally.
- **1** = a rendering or content problem prevents a reasonable judgment (for example, missing options, unreadable text, or a question that does not match the passage in a way that makes review impossible).

A difficult but readable item is `cannot_judge = 0`.

### 4.5 optional_stage1_note

Free text. Use it for a brief comment if needed. It is optional.

## 5. Stage 2 codes

The official key is now visible. Code four binary concerns. Mark 1 only when the stated definition is met. Difficulty alone is not a reason to mark 1.

### 5.1 key_problem

**1** = the official key is not uniquely supported, another option is also reasonably correct, or the official key lacks sufficient support from the item.

**0** = the official key is reasonably supported as the best answer.

Do not mark 1 merely because the question is difficult.

Example of 1: The passage supports both B and D equally, but the key is B. Example of 0: The key is C, the passage states C clearly, and the other options conflict with the passage, even if the wording is demanding.

Boundary: If you personally chose a different option in Stage 1, that does not automatically mean the key is wrong. Re-read the item with the key in view and ask whether the key is reasonably the best answer, not whether it was easy.

### 5.2 stem_problem

**1** = the question wording is meaningfully unclear, allows more than one reasonable interpretation, or omits information needed to understand what is being asked.

**0** = the question is clear enough to support a reasonable answer.

Long wording, advanced vocabulary, or difficult reasoning alone is not a stem problem.

Example of 1: The question says “Which is true?” when two opposite readings of “true” (true in the story vs. true in general) are both reasonable and lead to different options. Example of 0: The question is long and uses uncommon words, but it is clear that the reader must choose the best title.

### 5.3 distractor_problem

**1** = at least one distractor has a clear design problem that could affect interpretation or the quality of the item, such as: duplicate or near-duplicate choices; strong overlap with the keyed answer; a grammatical or formal clue that makes an option trivially removable; or an obviously unrelated or structurally inconsistent option.

**0** = no clear distractor-design concern is present.

A distractor is not automatically problematic merely because a knowledgeable person can eliminate it.

Example of 1: Options B and C are the same sentence with one extra comma, or three options are complete sentences while one is a fragment that cannot fit the blank. Example of 0: A distractor restates a detail from the wrong paragraph and can be eliminated by careful reading.

### 5.4 evidence_problem

**1** = the passage does not provide enough evidence for the keyed answer, provides conflicting evidence, or answering correctly requires important outside knowledge that the item does not make clear.

**0** = the passage provides enough information for the keyed answer.

Evidence distributed across several parts of the passage is not, by itself, an evidence problem.

Example of 1: The key requires a fact never stated or implied, or two sentences directly contradict each other and the key depends on choosing one without guidance. Example of 0: The reader must combine a rule in paragraph 1 with an example in paragraph 4. That is distributed evidence, not an evidence problem.

### 5.5 optional_stage2_note

Write a short explanation when any of the four concerns is coded 1. The note should name the concern and the textual reason. Keep it brief.

## 6. What is computed later

The research team will compute `overall_review_flag = 1` if any of `key_problem`, `stem_problem`, `distractor_problem`, or `evidence_problem` equals 1; otherwise 0. Do not enter this flag yourself.

## 7. Boundary cases

- Hard reasoning + supported key = no key problem.
- Distributed but sufficient evidence + supported key = no evidence problem.
- Advanced vocabulary + clear task = no stem problem.
- A well-written wrong option that a careful reader can reject = no distractor problem.
- Two options both fully supported = key problem.
- The question can be read in two incompatible ways that change the answer = stem problem.
- Two options are near-duplicates = distractor problem.
- The key depends on a fact the passage never supplies = evidence problem, and often also a key problem. Code each definition separately.

## 8. Independence, confidentiality, and process

- Code independently. Do not discuss individual formal items with the other coder.
- Do not try to infer any grouping, model output, or study hypothesis.
- Do not look up answers from outside sources.
- Use only the passage, question, options, and, in Stage 2, the official key.
- After the pilot, wording questions may be discussed. The codebook is then frozen. Formal coding uses the frozen codebook only.
