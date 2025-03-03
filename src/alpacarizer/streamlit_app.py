import streamlit as st
import json
import os
from google import genai
from typing import List, Dict, Any

# Configure page settings
st.set_page_config(page_title="Instruction Tuple Generator", layout="wide")

# Initialize session state variables if they don't exist
if 'candidates' not in st.session_state:
    st.session_state.candidates = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'output_file' not in st.session_state:
    st.session_state.output_file = "instruction_data.json"
if 'saved_data' not in st.session_state:
    # Try to load existing data if available
    if os.path.exists(st.session_state.output_file):
        try:
            with open(st.session_state.output_file, 'r') as f:
                st.session_state.saved_data = json.load(f)
        except:
            st.session_state.saved_data = []
    else:
        st.session_state.saved_data = []

# Function to generate instruction-response tuples using Gemini API
def generate_candidates(text: str, api_key: str) -> List[Dict[str, str]]:
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=api_key)
        
        # Construct the prompt
        prompt = f"""
        Based on the following text:
        
        {text}
        
        Generate 10 instruction-response tuples in the style of the Alpaca dataset for fine-tuning language models.
        Each tuple should contain an instruction related to the text and a corresponding response.
        Format the output as a JSON array with objects containing 'instruction' and 'response' keys.
        Do not include any explanation or conversation, just return valid JSON that can be parsed.
        """
        
        # Generate content with Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
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
    
    except Exception as e:
        st.error(f"Error generating candidates: {str(e)}")
        return []

# Function to save current candidate to file
def save_current_candidate():
    if not st.session_state.candidates:
        st.warning("No candidates to save.")
        return
    
    # Get the current candidate with any edits
    current_candidate = {
        "instruction": st.session_state.current_instruction,
        "response": st.session_state.current_response
    }
    
    # Append to saved data
    st.session_state.saved_data.append(current_candidate)
    
    # Save to file
    try:
        with open(st.session_state.output_file, 'w') as f:
            json.dump(st.session_state.saved_data, f, indent=2)
        st.success(f"Saved candidate to {st.session_state.output_file}")
    except Exception as e:
        st.error(f"Error saving to file: {str(e)}")

# Main app layout
st.title("Instruction Tuple Generator")

# API Key input
api_key = st.text_input("Enter your Gemini API Key:", type="password")

# File configuration
output_file = st.text_input("Output JSON file name:", value=st.session_state.output_file)
if output_file != st.session_state.output_file:
    st.session_state.output_file = output_file

# Text input area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Text")
    input_text = st.text_area(
        "Enter the text context for generating instruction-response tuples:",
        height=300
    )

    # Generate button
    if st.button("Generate Instruction-Response Tuples"):
        if not api_key:
            st.error("Please enter your Gemini API Key.")
        elif not input_text:
            st.error("Please enter some text to generate tuples from.")
        else:
            with st.spinner("Generating instruction-response tuples..."):
                candidates = generate_candidates(input_text, api_key)
                if candidates:
                    st.session_state.candidates = candidates
                    st.session_state.current_index = 0
                    # Initialize the first candidate in the text areas
                    if len(candidates) > 0:
                        st.session_state.current_instruction = candidates[0].get("instruction", "")
                        st.session_state.current_response = candidates[0].get("response", "")
                    st.success(f"Generated {len(candidates)} instruction-response tuples.")
                else:
                    st.error("Failed to generate valid instruction-response tuples.")

with col2:
    st.subheader("Statistics")
    st.write(f"Total candidates generated: {len(st.session_state.candidates)}")
    st.write(f"Total candidates saved: {len(st.session_state.saved_data)}")
    
    if os.path.exists(st.session_state.output_file):
        st.write(f"Output file exists: {st.session_state.output_file}")
    else:
        st.write("Output file will be created upon saving.")

# Candidate viewer and editor
st.subheader("Candidate Viewer and Editor")

if st.session_state.candidates:
    # Navigation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("Previous"):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
    
    with col2:
        current_idx = st.slider(
            "Navigate candidates:", 
            min_value=0, 
            max_value=len(st.session_state.candidates)-1, 
            value=st.session_state.current_index,
            key="candidate_slider"
        )
        if current_idx != st.session_state.current_index:
            st.session_state.current_index = current_idx
    
    with col3:
        if st.button("Next"):
            if st.session_state.current_index < len(st.session_state.candidates) - 1:
                st.session_state.current_index += 1
    
    # Display and edit current candidate
    if 0 <= st.session_state.current_index < len(st.session_state.candidates):
        candidate = st.session_state.candidates[st.session_state.current_index]
        
        # Initialize session state variables for the text areas if not present
        if 'current_instruction' not in st.session_state:
            st.session_state.current_instruction = candidate.get("instruction", "")
        if 'current_response' not in st.session_state:
            st.session_state.current_response = candidate.get("response", "")
        
        # Update text areas when index changes
        if st.session_state.current_index != getattr(st, 'last_index', None):
            st.session_state.current_instruction = candidate.get("instruction", "")
            st.session_state.current_response = candidate.get("response", "")
            setattr(st, 'last_index', st.session_state.current_index)
        
        # Editable text areas
        st.text_area(
            "Instruction:",
            value=st.session_state.current_instruction,
            height=150,
            key="instruction_editor"
        )
        
        st.text_area(
            "Response:",
            value=st.session_state.current_response,
            height=250,
            key="response_editor"
        )
        
        # Save button
        if st.button("Save Current Candidate"):
            save_current_candidate()
else:
    st.info("Generate instruction-response tuples to view and edit candidates.")

# Show saved data
if st.session_state.saved_data:
    with st.expander("View Saved Data"):
        st.json(st.session_state.saved_data)