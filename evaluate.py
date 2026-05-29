# Import module dependencies
from dotenv import load_dotenv
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import WhitespaceTokenizer

# Load environmental variables
load_dotenv()

# Load tokenizer for human and LLM translations
tokenizer = WhitespaceTokenizer

# Store corpora parent path into accessible variable
parent_file_path = os.getenv("CORPORA_PARENT_PATH")

# Defining weights for uni-gram, bi-gram. tri-gram and 4-gram
weights = (0.25, 0.25, 0, 0)

# Human translation file path
human_file_path = parent_file_path + ("/English/Decade_Of_Sheng_Min_Human.txt")

# Initialize lists to iterate over each line of human text
human_text = []

# Read and store each line from human text
with open(human_file_path, "r") as human:
    human_text = human.read().splitlines()

# Tokenize each line in human text
with open(human_file_path, "r") as source:
    for line in source:
        human_text.append(tokenizer.tokenize(line))

# LLM translation file path
llm_file_path = parent_file_path + ("/English/Decade_Of_Sheng_Min_Auto.txt")

# Initialize lists to iterate over each line of llm text
llm_text = []

# Tokenize each line in llm text
with open(llm_file_path, "r") as source:
    for line in source:
        llm_text.append(tokenizer.tokenize(line))

score = sentence_bleu(human_text, llm_text, weights=weights)
print(score)