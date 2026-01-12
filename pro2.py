

import os
import re
import json
import pdfplumber
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from google import genai

import pytesseract
from pdf2image import convert_from_path


# ENV

load_dotenv()


# CONFIG

PDF_PATH = r"D:\cvliq\florida_pro\florida_p1.pdf"
OUTPUT_CSV = "schema_extracted.csv"

START_PAGE = 1
END_PAGE = 25

SOURCE_URL = "https://floridasturnpike.com/wp-content/uploads/2025/12/TPK_PRESENTATION_OF_FY_2027_CONSULTANT_PLAN.pdf"
MODEL_NAME = "gemini-2.5-pro"

POPPLER_PATH = r"C:\Users\ADMIN\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# OCR

def ocr_page(pdf_path, page_number):
    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        poppler_path=POPPLER_PATH,
        use_pdftocairo=True
    )
    return pytesseract.image_to_string(images[0], config="--psm 6")


# CHUNKING

def build_chunks(text, page_no, max_chars=2000):
    chunks = []
    buf = ""

    for line in text.splitlines():
        if line.strip():
            buf += line + " "

        if len(buf) >= max_chars:
            chunks.append({"page": page_no, "text": buf.strip()})
            buf = ""

    if buf.strip():
        chunks.append({"page": page_no, "text": buf.strip()})

    return chunks

# SAFE JSON PARSER

def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return []

    return []


# NORMALIZE LIST FIELDS

def normalize_rows(rows):
    for r in rows:
        for k, v in r.items():
            if isinstance(v, list):
                r[k] = ", ".join(str(x) for x in v)
    return rows


# PROCUREMENT PROMPT

SYSTEM_PROMPT = """
You are a procurement data extraction engine.

The input text comes from procurement forecasts, consultant plans, capital programs,
and contracting outlook documents published by public agencies.

The text may come from tables, bullet lists, or paragraphs.
Each procurement opportunity is described by a block of text.

Extract one JSON object per real procurement opportunity.

RULES:
- Only extract values explicitly written
- Do not guess or infer
- Do not create fake opportunities
- If a field is missing, use null
- Do not rewrite descriptions

Schema:
[
  {
    "agency": string | null,
    "opportunity_source_type": string | null,
    "division": string | null,
    "contract_number": string | null,
    "expected_rfp_date": string | null,
    "expected_rfp_year": number | null,
    "expected_rfp_month": string | null,
    "min_contract_value": number | null,
    "max_contract_value": number | null,
    "contract_tags": string | null,
    "contract_type": string | null,
    "contract_term": string | null,
    "short_description": string | null,
    "detailed_description": string | null,
    "key_contact": string | null,
    "incumbent": string | null,
    "procurement_method": string | null,
    "location": string | null,
    "source_url": null,
    "status": string | null
  }
]

Output ONLY valid JSON.
"""

# GEMINI CALL

def map_chunk_with_gemini(chunk):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[SYSTEM_PROMPT, f"TEXT:\n{chunk['text']}"]
    )
    return safe_json_parse(response.text)


# PIPELINE

def run_pipeline():
    all_rows = []

    with pdfplumber.open(PDF_PATH) as pdf:
        for page_no in range(START_PAGE, END_PAGE + 1):
            page = pdf.pages[page_no - 1]

            text = page.extract_text()
            if not text or len(text.strip()) < 50:
                text = ocr_page(PDF_PATH, page_no)

            if not text:
                continue

            chunks = build_chunks(text, page_no)

            for chunk in chunks:
                rows = map_chunk_with_gemini(chunk)

                for r in rows:
                    r["source_url"] = SOURCE_URL

                all_rows.extend(rows)

    return all_rows


# SAVE
def save_output(rows):
    if not rows:
        print("No records extracted")
        return

    rows = normalize_rows(rows)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Extracted {len(df)} rows")
    print(f"Saved to {OUTPUT_CSV}")


# MAIN

if __name__ == "__main__":
    data = run_pipeline()
    save_output(data)
