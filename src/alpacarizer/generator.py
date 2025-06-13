import json
import time
from typing import List, Dict, Union
from markitdown import MarkItDown
from google import genai


class AlpacaGenerator:
    """
    Generates and evaluates Alpaca-style dataset entries from a source file using an LLM.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("An API key is required.")
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)
        self.text_converter = MarkItDown()

    def _call_llm(self, prompt: str) -> Union[List, Dict]:
        """Calls the generative model with retries and parses the JSON response."""
        for i in range(3):
            try:
                print(f"Attempt {i + 1} to call the LLM.")
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                response_text = response.text.strip()
                # print(f"Response received: {response_text}...")  # Log first 100 chars
                if "```json" in response_text:
                    json_text = (
                        response_text.split("```json")[1].split("```")[0].strip()
                    )
                else:
                    json_text = response_text

                return json.loads(json_text)
            except Exception as e:
                print(f"Error during Gemini API call on attempt {i + 1}: {e}")
                time.sleep(5)
        return []

    def evaluate_and_enrich(
        self, candidates: List[Dict[str, str]], original_text: str
    ) -> List[Dict[str, str]]:
        """
        Evaluates a list of candidates in batches and adds the evaluation scores to each candidate.
        """
        if not candidates:
            return []

        print(f"\n--- Starting evaluation of {len(candidates)} candidates ---")

        # Add a unique ID to each candidate for tracking during the API call
        for i, cand in enumerate(candidates):
            cand["id"] = i

        batch_size = 20
        all_evaluations_by_id = {}

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1}...")

            # Format the batch for the prompt, mapping instruction->question, output->answer
            eval_prompt_input = {
                "pairs": [
                    {
                        "id": p["id"],
                        "instruction": p["instruction"],
                        "output": p["output"],
                        "input": p.get("input", ""),
                    }
                    for p in batch
                ]
            }

            prompt = f"""You are a meticulous and highly critical Quality Assurance Inspector.
Your primary goal is to find flaws, inconsistencies, and weaknesses in a dataset of question-answer pairs.
You are evaluating data for a world-class AI, and the standards are exceptionally high. Most pairs will NOT be perfect.

Your task is to be as discerning as possible. Do not hesitate to give low scores for even minor issues. You must differentiate between merely "acceptable" pairs and "truly exceptional" ones.
You are an expert AI data curator, performing a batch evaluation of question-answer pairs.

First, you will analyze the provided text to identify the core topics and auxiliary topics. Then, you will evaluate each question-answer pair against a strict rubric.

**Text Analysis Instructions:**
1.  **Identify Core Topics:** Based on the content, identify and list the 5-10 most critical, recurring, and central themes, procedures, or concepts. These should be the topics most essential for a person to understand to ensure child safety.
2.  **Identify Auxiliary Topics:** Identify and list topics that are structural, meta-level, or administrative in nature. These are topics about the document itself, not the core mission of child protection.

** Context:**
{original_text}

Your task is to analyze the following JSON list of alpaca dataset-style instruction/(input)/output pairs. For EACH pair in the list, you must evaluate it against the provided rubric.

**Evaluation Rubric:**
1.  **Accuracy (Score 1-5, 1 -> incorrect, 3 -> correctness may be doubted, 5 -> correctness can be confirmed):** Is the answer factually correct?
2.  **Depth (Score 1-5, 1-> simple reproduction of a portion of the text, 5 -> transfer of knowledge neccessary):** How complex is the question? Extracting keywords, summarizing, or paraphrasing should be scored 4. Questions that require synthesis of multiple concepts or deeper understanding should be scored 5. Mere reproduction of a portion of the text should be scored 1.
3.  **Clarity (Score 1-5):** Are the question and answer well-phrased and self-contained, i.e. all information to answer the question is given in the provided context and the question does not refer to the text itsself (1 -> use of phases like "im vorliegenden Beitrag" or similar if no input key is present or the value of input key is empty)?
4.  **Relevance (Score 1-5):** Is the question relevant to the content or main theme of the provided text? (1 -> a question about the license of a text, about references or authors, 5 -> a question mentioning the main topic of the provided context)






1. Accuracy
    - Core Question: How factually correct is the provided answer when compared only to the source text?
    - Score 5: Perfectly Accurate
        - Explanation: The answer is a complete and precise reflection of the information present in the source document. Every fact, figure, and procedural step mentioned is entirely correct.
        - Example:
            - Question: "What is the mandatory timeframe for submitting a written report after an initial oral report?"
            - Answer: "A written report must be submitted within 48 hours of the initial oral report." (Assuming the text states this exactly).
    - Score 4: Substantially Accurate with Minor Omission
        - Explanation: The answer is factually correct in what it states, but it omits a minor, non-critical detail that would have made it more complete. The answer is not misleading.
        - Example:
            - Question: "What are the key components of a written report?"
            - Answer: "The report must include the child's name, age, and a description of the suspected abuse." (The answer is correct, but the source text also listed "date of observation" as a required component).
    - Score 3: Partially Accurate / Contains Minor Inaccuracy
        - Explanation: The answer is mostly correct but contains a specific factual error that, while not completely invalidating the answer, is significant.
        - Example:
            - Question: "What is the mandatory timeframe for submitting a written report?"
            - Answer: "A written report must be submitted within 72 hours of the initial oral report." (The source text says 48 hours. The answer addresses the right concept but gets the critical detail wrong).
    - Score 2: Substantially Inaccurate / Misleading
        - Explanation: The core of the answer is factually wrong or presents the information in a way that would lead a user to take the wrong action. It fundamentally misrepresents a key concept or procedure.
        - Example:
            - Question: "What is the first step upon suspecting abuse?"
            - Answer: "The first step is to schedule a meeting with the child's parents to discuss your concerns before contacting the authorities." (This is dangerously incorrect if the protocol demands immediate reporting to authorities).
    - Score 1: Completely Inaccurate / Contradictory
        - Explanation: The answer is demonstrably false and directly contradicts the source text.
        - Example:
            - Question: "Is written consent from a supervisor required to file a report?"
            - Answer: "No, written consent is not required from a supervisor." (The source text explicitly states that a supervisor's signature is mandatory).

2. Depth
    - Core Question: How much cognitive effort (recall, connection, inference, analysis) is required to answer this question from the source text?
    - Score 5: Synthesis & Analysis
        - Explanation: The question demands a high-level understanding of the entire document. The answer requires synthesizing concepts from multiple, distant chapters or analyzing competing principles or themes.
        - Example:
            - Question: "Compare and contrast the intervention strategies for 'at-risk families' versus 'confirmed-harm cases', explaining how the agency's core principles of 'child-centricity' and 'family preservation' are balanced differently in each."
            - Answer: (An answer that discusses how one strategy prioritizes support and resources while the other prioritizes immediate safety and legal action, linking both back to the book's stated core principles).
    - Score 4: Inference
        - Explanation: The answer is not explicitly stated but must be inferred by understanding the implications of the text, cause-and-effect relationships, or the author's intent. It requires "reading between the lines."
        - Example:
            - Question: "Why was the 'Immediate Contact' rule emphasized in the 2023 revision of the protocol?"
            - Answer: "The emphasis was likely added in response to incidents mentioned in the foreword where reporting delays led to further harm, highlighting the need to close procedural loopholes."
    - Score 3: Connecting Ideas
        - Explanation: The answer is not available in a single location. It requires the user to connect at least two distinct but related pieces of information from within the same section or chapter.
        - Example:
            - Question: "Based on the definition of 'emotional harm' and the reporting guidelines, when would a pattern of verbal insults warrant a report?"
            - Answer: "A report would be warranted when the pattern of insults, as described in the guidelines, causes observable signs of distress like anxiety or withdrawal, which falls under the definition of 'emotional harm'."
    - Score 2: Simple Recall
        - Explanation: The answer is explicitly stated in the text but may require reading a full paragraph or a list to assemble. It involves locating information but not connecting different ideas.
        - Example:
            - Question: "What are the five listed physical indicators of neglect?"
            - Answer: "The five indicators are poor hygiene, untreated medical issues, insufficient clothing, constant hunger, and lack of supervision."
    - Score 1: Trivial Lookup
        - Explanation: The answer can be found in a single, specific, and easily identifiable sentence (e.g., a definition in bold, a title). It requires minimal reading or comprehension.
        - Example:
            - Question: "What is the official name of the reporting protocol described in Chapter 3?"
            - Answer: "The protocol is named the 'Safe Harbor Reporting Standard'."

3. Clarity
    - Core Question: How well-phrased, grammatically correct, and unambiguous are the question and answer?
    - Score 5: Perfectly Clear
        - Explanation: Both question and answer use precise, professional language. They are grammatically perfect and contain no ambiguity, spelling errors, or awkward phrasing.
        - Example: "What are the legal responsibilities of a mandated reporter under section 4.1 of the code?"
    - Score 4: Slightly Awkward or Verbose
        - Explanation: The language is understandable and grammatically correct, but it may be slightly clunky, overly wordy, or use unnatural phrasing. The meaning is clear, but the delivery could be improved.
        - Example: "In the event that a report needs to be made, what is the thing that a person who has to report is supposed to do first?"
    - Score 3: Minor Ambiguity or Grammatical Errors
        - Explanation: Contains grammatical errors, spelling mistakes, or ambiguous phrasing that forces the reader to pause and re-read to understand the intended meaning.
        - Example: "When they talk to the child, what do they need to document about them?" (The pronoun "they" is ambiguous: who is "they"? The reporter? The authorities?).
    - Score 2: Substantially Unclear
        - Explanation: The question or answer is poorly structured to the point where it is very difficult to understand. It may contain multiple severe grammatical errors.
        - Example: "Reporting procedures what is about if child abuse is what you saw how to file it?"
    - Score 1: Incoherent
        - Explanation: The text is nonsensical, ungrammatical gibberish, or completely unrelated to the topic.
        - Example: "File why report no when see safe child is time always?"

4. Relevance
    - Core Question: How important is this question to the core mission of child protection, as defined by the source document?
    - Score 5: Centrally Relevant
        - Explanation: The question directly addresses a critical safety procedure, a legal obligation, or a primary definition of harm that is essential for protecting a child.
        - Example: "What immediate actions must be taken if a child is assessed to be in imminent danger?"
    - Score 4: Highly Relevant
        - Explanation: The question pertains to an important, standard process, a significant characterization of roles, or a key support mechanism. It's a crucial part of the broader system of child protection.
        - Example: "What long-term support services are available to families after an investigation is closed?"
    - Score 3: Moderately Relevant
        - Explanation: The question addresses standard operational context, preventative measures, or administrative requirements that support the core mission but are not direct, front-line procedures.
        - Example: "What are the annual training and certification requirements for staff members?"
    - Score 2: Marginally Relevant
        - Explanation: The question deals with background information, organizational history, or minor administrative details that have little direct impact on procedural execution.
        - Example: "In what year was the child protection agency originally founded?"
    - Score 1: Irrelevant
        - Explanation: The question is about a meta-detail of the document itself (e.g., its license, author, formatting) and has no bearing on the principles or procedures of child protection.
        - Example: "Under what license is this document published?"






**Your Task:**
Iterate through each object in the "pairs" list. Your output MUST be a single JSON object containing a key "evaluations". This key should hold a list of JSON objects. Each object in your output list must contain the original "id" and your evaluation scores.

**Example Output Structure (Your response must be ONLY this JSON):**
{{
  "evaluations": [
    {{
      "id": 101,
      "scores": {{ "accuracy": 5, "depth": 4, "clarity": 5, "relevance": 2 }},
      "overall_score": 4.0,
      "reasoning": "Excellent synthesis of concepts, but low relevance to the text."
    }}
  ]
}}

To calibrate your judgment, here are some examples of how to score pairs:

**--- Calibration Examples ---**

**Example 1: High Quality, High Relevance**
* **Input Pair:** {{ "id": 101, "question": "What is the first step a mandated reporter must take if they suspect emotional abuse?", "answer": "According to the protocol on page 12, the first step is to immediately contact the regional child protection hotline and provide a detailed account of their observations without notifying the family first." }}
* **Your Example Evaluation:**
    {{ "id": 101, "scores": {{ "accuracy": 5, "depth": 3, "clarity": 5, "relevance": 5 }}, "overall_score": 4.5, "reasoning": "Excellent. The question addresses a critical, high-relevance procedure. The answer is accurate and specific." }}

**Example 2: Accurate but Irrelevant (Your Specific Problem)**
* **Input Pair:** {{ "id": 102, "question": "Under what license is this document distributed?", "answer": "This document is distributed under the Creative Commons Attribution 4.0 International license." }}
* **Your Example Evaluation:**
    {{ "id": 102, "scores": {{ "accuracy": 5, "depth": 1, "clarity": 5, "relevance": 1 }}, "overall_score": 3.0, "reasoning": "While accurate, the question is completely irrelevant to the core mission of child protection. This is a low-value pair as it falls under the specified 'Low-Relevance Topics'." }}

**Example 3: Factually Incorrect**
* **Input Pair:** {{ "id": 103, "question": "What is the maximum time allowed to file a written report after an oral report?", "answer": "Mandated reporters are given up to 7 days to file a written report." }}
* **Your Example Evaluation:**
    {{ "id": 103, "scores": {{ "accuracy": 1, "depth": 2, "clarity": 5, "relevance": 5 }}, "overall_score": 3.25, "reasoning": "The question is highly relevant, but the answer is factually incorrect according to the manual, which specifies a 48-hour limit. This pair is unusable." }}

Now, please process the following batch of pairs with the same level of critical judgment:

**Input:**
{json.dumps(eval_prompt_input, indent=4)}
"""
            eval_response = self._call_llm(prompt)
            evaluations = (
                eval_response.get("evaluations", [])
                if isinstance(eval_response, dict)
                else []
            )

            # Store all evaluations from the batch in a central dictionary
            for e in evaluations:
                all_evaluations_by_id[e["id"]] = e

        # Enrich the original candidates list with the evaluation results
        for cand in candidates:
            evaluation = all_evaluations_by_id.get(cand["id"])
            if evaluation:
                # Add the evaluation results under a new key
                cand["evaluation"] = {
                    "scores": evaluation.get("scores"),
                    "overall_score": evaluation.get("overall_score"),
                    "reasoning": evaluation.get("reasoning"),
                }
            # Remove the temporary id
            del cand["id"]

        print(
            f"--- Evaluation complete. Enriched {len(candidates)} candidates with scores. ---"
        )
        return candidates

    def _generate_overview(self, text: str) -> List[Dict[str, str]]:
        """Generates instruction-input-output tuples."""
        prompt = f"""You are an expert AI data generator. Your task is to create high-quality question-answer pairs for fine-tuning a language model. The model's purpose is to become an expert on the provided text.

From the following text excerpt, please generate 20 instruction-output tuples in the style of the Alpaca dataset for fine-tuning language models.
Ensure the questions are diverse and cover the following types:

1.  **Factual:** A question that can be answered directly from a specific sentence in the text.
2.  **Summarization:** A question that requires summarizing the main point of the excerpt.
3.  **Inferential:** A question about a character's motivation, a cause-and-effect relationship, or something that is implied but not explicitly stated.
4.  **Analytical:** A question that asks about the 'why' or 'how' of a situation, theme, or character action described in the text.
5.  **Hypothetical (Advanced):** A question that asks "What might have happened if..." based on the information in the text.

The answers must be comprehensive, detailed (more than 256 tokens), and solely based on the information within this provided text excerpt. Do not use any outside knowledge.
The question must be self-contained without referencing the text (i.e. do not use "according to the text" etc.).

Here is the text excerpt:
{text}

Format the output as a JSON array with objects containing 'instruction' and 'output' keys.
Do not include any explanation or conversation, just return valid JSON that can be parsed.
Use German language.
        """
        return self._call_llm(prompt)

    def _generate_inputs(self, text: str) -> List[Dict[str, str]]:
        """Generates instruction-input-output tuples."""
        prompt = f"""You are an expert AI data generator. Your task is to create high-quality data for fine-tuning a language model. The model's purpose is to become an expert on the provided text.

Generate 10 instruction-input-output tuples in the style of the Alpaca dataset for fine-tuning language models out of the text below.
Each tuple should contain an istruction of the following instructions categories
- summarize, e.g., Provide a concise one-sentence summary of the following text:
- keyword, e.g., Extract 3-5 main keywords or key phrases from the following text:
- title, e.g., Generate a short, engaging title for the following text:
- sentiment, e.g., Analyze the sentiment of the following text. Classify it as positive, negative, or neutral, and briefly explain your reasoning:
- paraphrase, e.g., Rewrite the following text in your own words, maintaining its core meaning:
the input text for the instruction, and the corresponding response.
The 'input' should be between 64 and 512 tokens long.
The 'output' should be a single string, markdown formated, and between 64 and 512 tokens long.
Format the response as a JSON array with objects containing 'instruction', 'input', and 'output' keys.
Do not include any explanation or conversation, just return valid JSON that can be parsed.
Use German language.

** Text: **
{text}
        """
        return self._call_llm(prompt)

    def _generate_questions(self, text: str) -> List[Dict[str, str]]:
        """Generates instruction-output (question-answer) tuples."""
        prompt = f"""You are an expert AI data generator. Your task is to create high-quality data for fine-tuning a language model. The model's purpose is to become an expert on the provided text.

Generate 10 instruction-output tuples in the style of the Alpaca dataset for fine-tuning language models out of the text below.
- 5 tuples should have the questions mention to answer briefly and give complete but concise answers (answers with up to 256 tokens).
- 5 tuples should be complete and elaborate answers to the questions (answers with more than 256 tokens, markdown formated).
Each tuple should contain a question as the instruction and the corresponding response as the output.
The question must be self-contained without referencing the text (e.g "according to the text") and should not require additional context to be answered.
Format the response as a JSON array with objects containing 'instruction', 'output' keys.
Do not include any explanation or conversation, just return valid JSON that can be parsed.
Use German language.

** Text: **
{text}
        """
        return self._call_llm(prompt)

    def generate_from_text(
        self, text: str, evaluate: bool = True
    ) -> List[Dict[str, str]]:
        """
        Generates and optionally evaluates Alpaca-style entries from raw text.
        """
        all_candidates = []
        text_size = len(text)
        chunk_size, overlap = 8192, 512
        n_chunks = text_size // chunk_size + 1
        chunk_size = text_size // n_chunks
        
        print(f"Generating overview from the entire text...")
        all_candidates += self._generate_overview(text)
        time.sleep(5)

        print(
            f"Processing text with {text_size} characters. Number of chunks: {n_chunks}. Chunk size: {chunk_size} characters, overlap: {overlap} characters."
        )
        for i in range(0, text_size, chunk_size):
            chunk = text[i : i + chunk_size + overlap]
            if not chunk.strip():
                continue

            print(f"\n--- Generating from chunk {i // chunk_size + 1} ---")
            inputs = self._generate_inputs(chunk)
            time.sleep(5)
            questions = self._generate_questions(chunk)
            time.sleep(5)

            chunk_candidates = (inputs or []) + (questions or [])
            print(f"Generated {len(chunk_candidates)} raw candidates from this chunk.")
            all_candidates.extend(chunk_candidates)

        if evaluate and all_candidates:
            return self.evaluate_and_enrich(all_candidates, text)
        return all_candidates

    def generate_from_file(
        self, file_path: str, evaluate: bool = True
    ) -> List[Dict[str, str]]:
        """
        Extracts text from a file, generates, and optionally evaluates entries.
        """
        try:
            result = self.text_converter.convert(file_path)
            text_content = result.text_content
            return self.generate_from_text(text_content, evaluate=evaluate)
        except Exception as e:
            print(f"Failed to process file {file_path}: {e}")
            return []
