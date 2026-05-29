# Import module dependencies
from dotenv import load_dotenv
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Load environmental variables
load_dotenv()

# Store corpora parent path into accessible variable
parent_file_path = os.getenv("CORPORA_PARENT_PATH")

# Defining weights for uni-gram, bi-gram. tri-gram and 4-gram
weights = (0.25, 0.25, 0, 0)

# Human translation file path
human_file_path = parent_file_path + ("/Ancient_Chinese/Decade_Of_Sheng_Min.txt")

# Initialize lists to iterate over each line of human text
human_text = []

# Read and store each line from human text
with open(human_file_path, "r") as human:
    human_text = human.read().splitlines()

# LLM translation file path
llm_file_path = parent_file_path + ("/Ancient_Chinese/Decade_Of_Sheng_Min.txt")

# Initialize lists to iterate over each line of llm text
llm_text = []

# Read and store each line from llm text
with open(llm_file_path, "r") as llm:
    llm_text = llm.read().splitlines()

score = sentence_bleu(human_text, llm_text, weights=weights)
print(score)