import json
import os
from typing import List, Dict, Union, Optional


class AlpacaContainer:
    """
    A class to manage an Alpaca-style dataset stored in a JSON file.

    This class handles loading a dataset (which is a list of dictionaries)
    from a JSON file into memory, appending new entries to it, and saving
    the updated dataset back to a file. The standard Alpaca format is a
    list of dictionaries, each with 'instruction', 'input', and 'output' keys.
    """

    def __init__(self, file_path: str = "alpaca_dataset.json"):
        """
        Initializes the AlpacaContainer.

        If the specified file exists, it loads the data from it. Otherwise,
        it starts with an empty dataset.

        Args:
            file_path (str): The path to the JSON file where the dataset
                             is stored. Defaults to "alpaca_dataset.json".
        """
        self.file_path: str = file_path
        self.data: List[Dict[str, str]] = []
        self._load()

    def _load(self):
        """
        Loads the dataset from the JSON file into the in-memory list.

        Handles cases where the file doesn't exist, is empty, or contains
        invalid JSON.
        """
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                    # Basic validation to ensure it's a list
                    if not isinstance(self.data, list):
                        print(
                            f"Warning: Data in {self.file_path} is not a list. Initializing an empty dataset."
                        )
                        self.data = []
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON from {self.file_path}. Initializing an empty dataset."
                )
                self.data = []
            except Exception as e:
                print(f"An unexpected error occurred while loading the file: {e}")
                self.data = []
        else:
            # If the file doesn't exist or is empty, start with an empty list.
            self.data = []

    def save(self, file_path: Optional[str] = None):
        """
        Saves the in-memory dataset to a JSON file.

        The JSON is saved in a human-readable (pretty-printed) format.

        Args:
            file_path (str, optional): The path to save the file to.
                                       If None, it uses the path provided
                                       during initialization.
        """
        save_path = file_path or self.file_path
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            print(f"Dataset successfully saved to {save_path}")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")

    def append(self, entry_or_entries: Union[Dict[str, str], List[Dict[str, str]]]):
        """
        Appends a new entry or a list of entries to the in-memory dataset.

        Args:
            entry_or_entries (Union[Dict, List[Dict]]): A single dictionary or a list
                                                        of dictionaries to append.
                                                        Each dict must have an 'instruction' key.
                                                        'input' and 'output' are optional.

        Raises:
            TypeError: If the provided argument is not a dict or a list of dicts.
            ValueError: If a dictionary is missing the 'instruction' key or it's empty.
        """
        if isinstance(entry_or_entries, dict):
            entries_to_add = [entry_or_entries]
        elif isinstance(entry_or_entries, list):
            entries_to_add = entry_or_entries
        else:
            raise TypeError("Argument must be a dictionary or a list of dictionaries.")

        validated_entries = []
        for entry in entries_to_add:
            if not isinstance(entry, dict):
                raise TypeError(
                    f"All items in the list must be dictionaries. Found: {type(entry)}"
                )

            if "instruction" not in entry or not entry["instruction"]:
                raise ValueError("Each entry must have a non-empty 'instruction' key.")

            # Ensure 'input' and 'output' keys exist for consistency
            if "input" not in entry:
                entry["input"] = ""
            if "output" not in entry:
                entry["output"] = ""

            validated_entries.append(entry)

        self.data.extend(validated_entries)

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Allows you to use the `len()` function on an instance of this class.
        """
        return len(self.data)

    def __str__(self) -> str:
        """
        Returns a user-friendly string representation of the container.
        """
        return f"AlpacaContainer(file_path='{self.file_path}', items={len(self)})"

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the object, useful for debugging.
        """
        return f"<AlpacaContainer file='{self.file_path}' len={len(self)}>"


# --- Example Usage ---
if __name__ == "__main__":
    dataset_file = "my_alpaca_dataset.json"
    # Clean up previous runs if file exists for a clean demonstration
    if os.path.exists(dataset_file):
        os.remove(dataset_file)

    # 1. Create a container instance.
    container = AlpacaContainer(file_path=dataset_file)
    print(f"Initial state: {container}")

    # 2. Append a single data point (as a dict)
    print("\nAppending a single data point...")
    single_entry = {
        "instruction": "Translate the following English text to French.",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment ça va ?",
    }
    container.append(single_entry)
    print(f"Dataset now contains {len(container)} item.")

    # 3. Append a list of data points
    print("\nAppending a list of data points...")
    list_of_entries = [
        {
            "instruction": "Summarize the following paragraph.",
            "input": "The quick brown fox jumps over the lazy dog. This sentence is famous because it contains all the letters of the English alphabet.",
            "output": "The sentence 'The quick brown fox jumps over the lazy dog' is well-known for using every letter in the alphabet.",
        },
        {
            "instruction": "What is the capital of Japan?",
            # 'input' key is missing, will be added automatically with an empty string
            "output": "Tokyo",
        },
    ]
    container.append(list_of_entries)

    # 4. Check the size of the dataset in memory
    print(f"\nDataset now contains {len(container)} items.")

    # 5. Save the data to the file
    print("\nSaving data to the file...")
    container.save()

    # 6. Create a new container instance to verify loading from the file
    print("\nCreating a new container to load the saved file...")
    new_container = AlpacaContainer(file_path=dataset_file)
    print(f"Loaded container state: {new_container}")
    print(f"It has {len(new_container)} items, which should match the saved data.")
    # print("Loaded data:", json.dumps(new_container.data, indent=2))

    # Clean up the created file
    if os.path.exists(dataset_file):
        os.remove(dataset_file)
        print(f"\nCleaned up the example file: {dataset_file}")
