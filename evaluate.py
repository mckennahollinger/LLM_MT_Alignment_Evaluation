# Import module dependencies
from dotenv import load_dotenv
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import WhitespaceTokenizer

# Load environmental variables
load_dotenv()

# Load tokenizer for human and LLM translations
tokenizer = WhitespaceTokenizer()

# Store corpora parent path into accessible variable
parent_file_path = os.getenv("CORPORA_PARENT_PATH")

# Human translation file path
human_file_path = parent_file_path + ("/English/Decade_Of_Sheng_Min_Human.txt")

# Initialize lists to iterate over each line of human text
human_text = []

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

# Initialize smoothing technique for sentence-level BLEU score by Boxing Chen and Collin Cherry (2014)
chencherry = SmoothingFunction()

# Score each sentence using Chen and Cherry's 2nd smoothing technique
for idx in range(0, len(human_text)):
    score = sentence_bleu(human_text[idx], llm_text[idx], smoothing_function=chencherry.method2)
    print(score)