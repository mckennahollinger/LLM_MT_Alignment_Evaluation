# Import module dependencies
from dotenv import load_dotenv
import os
from transformers import pipeline

# Load environmental variables
load_dotenv()

# Store corpora parent path into accessible variable
parent_file_path = os.getenv("CORPORA_PARENT_PATH")

# Source corpora file path
source_file_path = parent_file_path + ("/Ancient_Chinese/Decade_Of_Sheng_Min.txt")

# Open and read source text into string
source_text = open(source_file_path, "r")

# Load detection model
detection_model = "papluca/xlm-roberta-base-language-detection"


pipe = pipeline("text-classification", model=detection_model)

print(pipe(source_text, top_k=1, truncation=True))

