import os
from openai import OpenAI
import json
from tqdm import tqdm
from dotenv import load_dotenv
import re

load_dotenv()

# ----------------------------
# 1. OpenAI API client
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # or set directly

# ----------------------------
# 2. Read text and split paragraphs
# ----------------------------
def read_paragraphs(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split by double newlines (common for paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]  # skip tiny lines
    return paragraphs

# ----------------------------
# Helper: minimal cleaning for assistant content
# ----------------------------
def clean_text(s: str) -> str:
    """
    Minimal cleaning for the paragraph text to be used as assistant content:
    - Normalize CRLF to LF
    - Remove null bytes
    - Collapse multiple newlines to a single newline
    - Replace any remaining newlines with a single space (so JSONL assistant content is one-line)
    - Collapse multiple spaces into a single space
    - Strip leading/trailing whitespace
    """
    if not s:
        return s
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\x00", " ")
    # collapse many newlines into one
    s = re.sub(r'\n{2,}', '\n', s)
    # replace remaining newlines with space (so content isn't filled with \n in JSON)
    s = s.replace('\n', ' ')
    # collapse multiple whitespace into single space
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

# ----------------------------
# 3. Function to generate a question from a paragraph
#    (LEFT UNCHANGED as requested)
# ----------------------------
def generate_question(paragraph):
    messages = [
        {
            "role": "system",
            "content": "You are an expert in economics. Generate one clear, concise question based only on the paragraph provided."
        },
        {
            "role": "user",
            "content": paragraph
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",  # or gpt-3.5-turbo
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )
        question = response.choices[0].message.content.strip()
        return question
    except Exception as e:
        print("Error generating question:", e)
        return None

# ----------------------------
# 4. Main processing
# ----------------------------
def create_qa_dataset(txt_file, output_file):
    paragraphs = read_paragraphs(txt_file)
    qa_list = []

    for paragraph in tqdm(paragraphs):
        question = generate_question(paragraph)
        if question:
            # minimal clean of question for duplicate-checking / tidy output (DOES NOT change generation)
            clean_question = re.sub(r'\s{2,}', ' ', question).strip()
            # clean the paragraph only for writing to JSONL (assistant content)
            cleaned_paragraph = clean_text(paragraph)

            qa_list.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in economics. Answer the user's question using only your knowledge from the fine-tuned content."
                    },
                    {
                        "role": "user",
                        "content": clean_question
                    },
                    {
                        "role": "assistant",
                        "content": cleaned_paragraph
                    }
                ]
            })

    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in qa_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(qa_list)} Q/A pairs to {output_file}")

# ----------------------------
# 5. Example usage
# ----------------------------
if __name__ == "__main__":
    txt_file = "./dataset/pg3300.txt"        # your large text file
    output_file = "qa_big_dataset.jsonl"
    create_qa_dataset(txt_file, output_file)
