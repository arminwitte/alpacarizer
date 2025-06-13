from google import genai
from typing import List, Dict
import time
import json


def generator_call(prompt: str, api_key: str) -> List[Dict[str, str]]:
    def acquire_candidates(prompt: str, api_key: str) -> List[Dict[str, str]]:
        # Initialize the Gemini client
        client = genai.Client(api_key=api_key)
        # Generate content with Gemini
        response = client.models.generate_content(
            model="gemini-1.5-flash",  # "gemini-2.5-flash-preview-05-20", # "gemini-2.0-flash",
            contents=prompt,
        )

        # Extract and parse the JSON response
        response_text = response.text.strip()
        # Handle potential formatting issues by extracting just the JSON part
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text

        # Parse the JSON
        candidates = json.loads(json_text)
        return candidates

    candidates = []
    for _ in range(3):  # Retry up to 3 times
        try:
            candidates = acquire_candidates(prompt, api_key)
            break
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            time.sleep(5)  # wait to avoid rate limiting or transient errors
    return candidates


# Function to generate instruction-input-output tuples using Gemini API
def generate_candidates_input(text: str, api_key: str) -> List[Dict[str, str]]:
    # Construct the prompt
    prompt = f"""
    Based on the following text:
    
    {text}
    
    Generate 10 instruction-input-output tuples in the style of the Alpaca dataset for fine-tuning language models.
    Each tuple should contain an istruction of the following instructions categories
    - summarize, e.g., Provide a concise one-sentence summary of the following text:
    - keyword, e.g., Extract 3-5 main keywords or key phrases from the following text:
    - title, e.g., Generate a short, engaging title for the following text:
    - sentiment, e.g., Analyze the sentiment of the following text. Classify it as positive, negative, or neutral, and briefly explain your reasoning:
    - paraphrase, e.g., Rewrite the following text in your own words, maintaining its core meaning:
    the input text for the instruction, and the corresponding response.
    The input should be between 64 and 512 tokens long.
    Format the output as a JSON array with objects containing 'instruction', 'input', and 'output' keys.
    Do not include any explanation or conversation, just return valid JSON that can be parsed.
    Use German language.
    """

    return generator_call(prompt, api_key)


# Function to generate instruction-input-output tuples using Gemini API
def generate_candidates_questions(text: str, api_key: str) -> List[Dict[str, str]]:
    # Construct the prompt
    prompt = f"""
    Based on the following text:
    
    {text}
    
    Generate 10 instruction-output tuples in the style of the Alpaca dataset for fine-tuning language models.
    Each tuple should contain a question as the instruction and the corresponding response as the output.
    The question must be self-contained without referencing the text (e.g "according to the text") and should not require additional context to be answered.
    The output should be a complete and concise answer to the question.
    Format the output as a JSON array with objects containing 'instruction', 'output' keys.
    Do not include any explanation or conversation, just return valid JSON that can be parsed.
    Use German language.
    """

    return generator_call(prompt, api_key)


# Function to generate instruction-input-output tuples using Gemini API
def generate_candidates(text: str, api_key: str) -> List[Dict[str, str]]:
    inputs = generate_candidates_input(text, api_key)
    time.sleep(5)  # Sleep for 5 seconds to avoid rate limiting
    questions = generate_candidates_questions(text, api_key)
    time.sleep(5)  # Sleep for 5 seconds to avoid rate limiting
    return inputs + questions
