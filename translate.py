# Import module dependencies
from dotenv import load_dotenv
import os
from transformers import pipeline

# Load environmental variables
load_dotenv()

# Store corpora parent path into accessible variable
parent_file_path = os.getenv("CORPORA_PARENT_PATH")

# Load detection model
detection_model = "papluca/xlm-roberta-base-language-detection"

# Source corpora file path
source_file_path = parent_file_path + ("/Ancient_Chinese/Decade_Of_Sheng_Min.txt")

# Initialize lists to iterate over each line of text
source_text = []

# Read and store each line from source text
with open(source_file_path, "r") as source:
    source_text = source.read().splitlines()

# Create language detection pipeline
pipe = pipeline("text-classification", model=detection_model)

# Print results of language detected for each line
print(pipe(source_text, top_k=1, truncation=True))

