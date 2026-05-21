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
target_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Initialize lists to iterate over each tokenized line of text
source_tokens = []
target_tokens = []

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
target_file_path = parent_file_path + ("/English/Decade_Of_Sheng_Min.txt")

# Tokenize each line in target text
with open(target_file_path, "r") as target:
    for line in target:
        target_tokens.append(target_tokenizer.tokenize(line))

print(source_tokens)
print(target_tokens)

# Align source and target texts
alignments = aligner.get_word_aligns(source_tokens, target_tokens)

# Print alignment results
for s,t in alignments["itermax"]:
  print(f"{source_tokenizer[s]} ({s}) === {target_tokenizer[t]} ({t})")