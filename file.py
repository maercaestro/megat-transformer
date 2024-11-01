import os

# Define the base directory and structure
base_dir = '/Users/abuhuzaifahbidin/Documents/GitHub/attention-paper'
structure = {
    "src": ["__init__.py", "encoder.py", "decoder.py", "transformer.py", "utils.py", "attention.py"],
    "data": ["sample_data.csv"],
    "scripts": ["train.py", "evaluate.py"],
    "tests": ["__init__.py", "test_encoder.py", "test_decoder.py", "test_transformer.py", "test_utils.py"],
    "notebook" : ["tests.ipynb"],
    "config": "",
    "": ["requirements.txt", "README.md"]  # Root files
}

# Create directories and files
os.makedirs(base_dir, exist_ok=True)
for folder, files in structure.items():
    folder_path = os.path.join(base_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    for file in files:
        open(os.path.join(folder_path, file), 'a').close()  # Create each file

base_dir
