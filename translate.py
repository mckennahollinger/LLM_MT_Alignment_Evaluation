# Import module dependencies
from dotenv import load_dotenv
import os
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast

# Load environmental variables
load_dotenv()

'''
Intersection of languages supported for detection through xlm-roberta-base-language-detection and 
multilingual translation through mbart-large-50-many-to-many-mmt
'''
languageCodes = {"ar": "ar_AR", "de": "de_DE", "en": "en_XX", "es": "es_XX",  "fr": "fr_XX", "hi": "hi_IN", "it": "it_IT", 
                 "ja": "ja_XX", "nl": "nl_XX", "pl": "pl_PL", "pt": "pt_XX", "ru": "ru_RU", "sw": "sw_KE", "th": "th_TH", 
                 "tr": "tr_TR", "ur": "ur_PK", "vi": "vi_VN", "zh": "zh_CN"}

# Store corpora parent path into accessible variable
parent_file_path = os.getenv("CORPORA_PARENT_PATH")

# Load detection model
detection_model = "papluca/xlm-roberta-base-language-detection"

# Load translation model
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Load tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Source corpora file path
source_file_path = parent_file_path + ("/Ancient_Chinese/Decade_Of_Sheng_Min.txt")

# Initialize lists to iterate over each line of text
source_text = []

# Read and store each line from source text
with open(source_file_path, "r") as source:
    source_text = source.read().splitlines()

# Create language detection pipeline
pipe = pipeline("text-classification", model=detection_model)

# Store results of language detected for each line
detected = (pipe(source_text, top_k=1, truncation=True))

# Retrive language code for automatic translation and translate from detected language to English
for line, detection in enumerate(detected):
    for label in detection:
        tokenizer.src_lang = languageCodes[label['label']]
        tokenized = tokenizer(source_text[line], return_tensors="pt")
        generated_tokens = translation_model.generate(**tokenized, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
        print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))