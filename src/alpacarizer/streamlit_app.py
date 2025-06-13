import streamlit as st
import json
import os
from markitdown import MarkItDown
import tempfile

from alpacarizer.utils import generate_candidates
from alpacarizer.generator import AlpacaGenerator


# ============================================================================
# process several files at once
# ============================================================================


@st.dialog("Process File")
def process_files(api_key: str):
    # File upload
    uploaded_files = st.file_uploader(
        "Upload a document", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    def upload_file_to_api(uploaded_files, api_key: str):
        if uploaded_files:
            for file in uploaded_files:
                # Ensure the file is not empty
                if file is None or file.size == 0:
                    st.error("Please upload a valid file.")
                    return

                content = file.getvalue()
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(file.name)[1]
                ) as temp_file:
                    if isinstance(content, str):
                        temp_file.write(content.encode())
                    else:
                        temp_file.write(content)
                    temp_path = temp_file.name

                # try:
                #     md = MarkItDown()
                #     result = md.convert(temp_path)
                #     text = result.text_content
                # finally:
                #     if os.path.exists(temp_path):
                #         os.remove(temp_path)

                # candidates = []
                # chunk_size = 8192  # Define chunk size
                # overlap = 512
                # print(f"{file.name}:\nNumber of chunks: {len(text) // chunk_size + 1}")
                # for i in range(0, len(text), chunk_size):
                #     chunk = text[i : i + chunk_size + overlap]
                #     if not chunk.strip():
                #         continue
                #     candidates += generate_candidates(chunk, api_key)
                #     print(
                #         f"n candidates: {len(candidates)}, n saved_data: {len(st.session_state.saved_data)}"
                #     )

                # --- 1. Generate and Evaluate Data ---
                print("#############################################")
                print(f"Processing file: {file.name}")
                print("--- STEP 1: GENERATING AND ENRICHING DATA ---")
                try:
                    generator = AlpacaGenerator(api_key=api_key)
                    # Generate and enrich data from the dummy file
                    candidates = generator.generate_from_file(temp_path, evaluate=True)
                    print(
                        f"\nSuccessfully generated and enriched {len(candidates)} data points."
                    )
                except ValueError as e:
                    print(f"Error: {e}")
                    candidates = []

                # # --- 2. Store and Save Enriched Data ---
                # if candidates:
                #     print("\n--- STEP 2: STORING ENRICHED DATA ---")
                #     container = AlpacaContainer(file_path=dataset_file)
                #     container.append(enriched_data)
                #     print(f"Container now holds {len(container)} enriched items.")
                #     container.save()

                st.session_state.saved_data += candidates
                try:
                    with open(st.session_state.output_file, "w") as f:
                        json.dump(st.session_state.saved_data, f, indent=2)
                    st.success(f"Saved candidate to {st.session_state.output_file}")
                except Exception as e:
                    st.error(f"Error saving to file: {str(e)}")

    st.button(
        "Send to database",
        use_container_width=True,
        on_click=upload_file_to_api,
        args=(uploaded_files, api_key),
    )


# Configure page settings
st.set_page_config(page_title="Instruction Tuple Generator", layout="wide")

# Initialize session state variables if they don't exist
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "output_file" not in st.session_state:
    st.session_state.output_file = "instruction_data.json"
if "saved_data" not in st.session_state:
    # Try to load existing data if available
    if os.path.exists(st.session_state.output_file):
        try:
            with open(st.session_state.output_file, "r") as f:
                st.session_state.saved_data = json.load(f)
        except:
            st.session_state.saved_data = []
    else:
        st.session_state.saved_data = []


# Function to save current candidate to file
def save_current_candidate():
    if not st.session_state.candidates:
        st.warning("No candidates to save.")
        return

    # Get the current candidate without any edits
    current_candidate = {
        "instruction": st.session_state.current_instruction,
        "input": st.session_state.current_input,
        "output": st.session_state.current_output,
    }

    # overwrite with edits
    if "instruction_editor" in st.session_state:
        current_candidate["instruction"] = st.session_state.instruction_editor
    if "input_editor" in st.session_state:
        current_candidate["input"] = st.session_state.input_editor
    if "output_editor" in st.session_state:
        current_candidate["output"] = st.session_state.output_editor

    # Append to saved data
    st.session_state.saved_data.append(current_candidate)

    # Save to file
    try:
        with open(st.session_state.output_file, "w") as f:
            json.dump(st.session_state.saved_data, f, indent=2)
        st.success(f"Saved candidate to {st.session_state.output_file}")
    except Exception as e:
        st.error(f"Error saving to file: {str(e)}")


# Main app layout
st.title("Instruction Tuple Generator")

# API Key input
api_key = st.text_input("Enter your Gemini API Key:", type="password")

# File configuration
output_file = st.text_input(
    "Output JSON file name:", value=st.session_state.output_file
)
if output_file != st.session_state.output_file:
    st.session_state.output_file = output_file

with st.sidebar:
    button_process_file = st.button("Process document", use_container_width=True)
    if button_process_file:
        process_files(api_key)

# Text input area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Text")
    input_text = st.text_area(
        "Enter the text context for generating instruction-input-output tuples:",
        height=300,
    )

    # Generate button
    if st.button("Generate instruction-input-output Tuples"):
        if not api_key:
            st.error("Please enter your Gemini API Key.")
        elif not input_text:
            st.error("Please enter some text to generate tuples from.")
        else:
            with st.spinner("Generating instruction-input-output tuples..."):
                candidates = generate_candidates(input_text, api_key)
                if candidates:
                    st.session_state.candidates = candidates
                    st.session_state.current_index = 0
                    # Initialize the first candidate in the text areas
                    if len(candidates) > 0:
                        st.session_state.current_instruction = candidates[0].get(
                            "instruction", ""
                        )
                        st.session_state.current_input = candidates[0].get("input", "")
                        st.session_state.current_output = candidates[0].get(
                            "output", ""
                        )
                    st.success(
                        f"Generated {len(candidates)} instruction-input-output tuples."
                    )
                else:
                    st.error(
                        "Failed to generate valid instruction-input-output tuples."
                    )

with col2:
    st.subheader("Statistics")
    st.write(f"Total candidates generated: {len(st.session_state.candidates)}")
    st.write(f"Total candidates saved: {len(st.session_state.saved_data)}")

    if os.path.exists(st.session_state.output_file):
        st.write(f"Output file exists: {st.session_state.output_file}")
    else:
        st.write("Output file will be created upon saving.")

# Save all candidates button
if st.button("Save All Candidates", use_container_width=True):
    for candidate in st.session_state.candidates:
        st.session_state.saved_data.append(candidate)
    try:
        with open(st.session_state.output_file, "w") as f:
            json.dump(st.session_state.saved_data, f, indent=2)
        st.success(f"Saved all candidates to {st.session_state.output_file}")
    except Exception as e:
        st.error(f"Error saving to file: {str(e)}")

# Candidate viewer and editor
st.subheader("Candidate Viewer and Editor")

# Navigation controls
col1, col2, col3, col4 = st.columns([1, 2, 1, 1])

with col1:
    if st.button("Previous"):
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1

with col2:
    current_idx = st.slider(
        "Navigate candidates:",
        min_value=0,
        max_value=len(st.session_state.candidates) - 1,
        value=st.session_state.current_index,
        key="candidate_slider",
    )
    if current_idx != st.session_state.current_index:
        st.session_state.current_index = current_idx

with col3:
    if st.button("Next"):
        if st.session_state.current_index < len(st.session_state.candidates) - 1:
            st.session_state.current_index += 1

with col4:
    # Save button
    if st.button("Save Current Candidate"):
        save_current_candidate()
        if st.session_state.current_index < len(st.session_state.candidates) - 1:
            st.session_state.current_index += 1

# Display and edit current candidate
if 0 <= st.session_state.current_index < len(st.session_state.candidates):
    candidate = st.session_state.candidates[st.session_state.current_index]

    # Initialize session state variables for the text areas if not present
    if "current_instruction" not in st.session_state:
        st.session_state.current_instruction = candidate.get("instruction", "")
    if "current_input" not in st.session_state:
        st.session_state.current_input = candidate.get("input", "")
    if "current_output" not in st.session_state:
        st.session_state.current_output = candidate.get("output", "")

    # Update text areas when index changes
    if st.session_state.current_index != getattr(st, "last_index", None):
        st.session_state.current_instruction = candidate.get("instruction", "")
        st.session_state.current_input = candidate.get("input", "")
        st.session_state.current_output = candidate.get("output", "")
        setattr(st, "last_index", st.session_state.current_index)

    # Editable text areas
    st.text_area(
        "Instruction:",
        value=st.session_state.current_instruction,
        height=150,
        key="instruction_editor",
    )

    st.text_area(
        "Input:", value=st.session_state.current_input, height=150, key="input_editor"
    )

    st.text_area(
        "Output:",
        value=st.session_state.current_output,
        height=150,
        key="output_editor",
    )

# Show saved data
if st.session_state.saved_data:
    with st.expander("View Saved Data"):
        st.json(st.session_state.saved_data)
