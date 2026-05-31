# Import module dependencies
from dotenv import load_dotenv
import os
import sentencepiece
from transformers import AutoTokenizer
from simalign import SentenceAligner

# Load environmental variables
load_dotenv()

# Load tokenizers for source and target texts
source_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
human_target_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Initialize lists to iterate over each tokenized line of text
source_tokens = []
human_target_tokens = []

# Store corpora parent path into accessible variable
parent_file_path = os.getenv("CORPORA_PARENT_PATH")

# Load alignment model
alignment_model = "UGARIT/grc-por-alignment-v3"
aligner = SentenceAligner(model=alignment_model, token_type="bpe", matching_methods="i")

# Source corpora file path
source_file_path = parent_file_path + ("/Ancient_Chinese/Decade_Of_Sheng_Min.txt")

# Tokenize each line in source text
with open(source_file_path, "r") as source:
    for line in source:
        source_tokens.append(source_tokenizer.tokenize(line))

# Target corpora file path
human_target_file_path = parent_file_path + ("/English/Decade_Of_Sheng_Min_Human.txt")

# Tokenize each line in target text
with open(human_target_file_path, "r") as target:
    for line in target:
        human_target_tokens.append(human_target_tokenizer.tokenize(line))

# Initialize list to append SimAlign Itermax algorithm results to 
alignments = []

# Align source and target lines of text 
for line in range(len(source_tokens)):
    alignments.append(aligner.get_word_aligns(source_tokens[line], human_target_tokens[line]))

# Print results of which words in each text have been aligned with each other
for line, aligned in enumerate(alignments):
    for pair in aligned["itermax"]:
        print(f"{source_tokens[line][pair[0]]} ({pair[0]}) === {human_target_tokens[line][pair[1]]} ({pair[1]})")