# UNIVERSAL PDF → TABLE-AWARE → GEMINI → STRICT JSON → CSV

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

PDF_PATH = r"D:\cvliq\newpoer_pdf\newport4.pdf"
OUTPUT_CSV = "schema_extracted.csv"

START_PAGE = 1
END_PAGE = 3

AGENCY = "New Jersey Turnpike Authority"
ASSET_TYPE = "Toll Road"
SOURCE_URL = "https://www.njta.gov/document/njta-traffic-revenue/"

MODEL_NAME = "gemini-2.5-pro"

POPPLER_PATH = r"C:\Users\ADMIN\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# OCR FALLBACK

def ocr_page(pdf_path: str, page_number: int) -> str:
    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        poppler_path=POPPLER_PATH,
        use_pdftocairo=True
    )
    return pytesseract.image_to_string(images[0], config="--psm 6")


# TABLE → STRUCTURED TEXT

def tables_to_text(tables):
    blocks = []
    for t in tables:
        df = pd.DataFrame(t)
        df = df.replace("\n", " ", regex=True)
        blocks.append(df.to_csv(index=False))
    return "\n\n".join(blocks)


# ROW-BASED CHUNKING (CRITICAL)

def build_chunks(text: str, page_no: int, max_rows=30):
    lines = text.splitlines()
    chunks, buf = [], []

    for line in lines:
        if line.strip():
            buf.append(line)

        if len(buf) >= max_rows:
            chunks.append({"page": page_no, "text": "\n".join(buf)})
            buf = []

    if buf:
        chunks.append({"page": page_no, "text": "\n".join(buf)})

    return chunks

# SAFE JSON PARSER

def safe_json_parse(text: str) -> List[Dict]:
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


# EXTRACTION PROMPT

SYSTEM_PROMPT = """
You are a financial table extraction engine for toll roads, bridges, tunnels, and similar transportation assets.

The input text comes from PDF tables that report financial and traffic data across multiple years, often in comparison format.
Layouts and column order may change between documents, but the meaning of the data does not.

Your job is to extract normalized financial facts.

────────────────────────────────────
STEP 1 — Identify the asset
────────────────────────────────────
From the document, infer:
• asset_type (toll road, bridge, tunnel, expressway, managed lanes, ferry, or other)
• asset (facility name if present)
• agency (operating authority if present)

ASSET TYPE NORMALIZATION:
The asset_type must be one of:
Toll Road, Bridge, Tunnel, Expressway, Managed Lanes, Ferry, Other.

Map similar terms as follows:
• Turnpike → Toll Road
• Tollway → Toll Road
• Parkway → Expressway
• Causeway → Bridge
• Crossing → Bridge
• Tunnel → Tunnel

If the asset type cannot be determined, use Other.
Do not invent new asset_type values.

────────────────────────────────────
STEP 2 — Discover table structure
────────────────────────────────────
The tables contain only two metrics:
• Revenue (money collected)
• Transactions (vehicle or trip counts)

Column labels such as “vehicles”, “traffic”, “trips”, or similar must be mapped to:
→ Transactions

Column labels such as “toll revenue”, “revenue”, “tolls”, or similar must be mapped to:
→ Revenue

The tables are broken down by:
• Year
• Period (month, quarter, or total)
• Segment (e.g., All Vehicles, Passenger Vehicles, Commercial Vehicles)

────────────────────────────────────
STEP 3 — Extract facts
────────────────────────────────────
For every valid combination of:
(year, period_reported, metric_type, segment)

extract one numeric value.

Tables may show multiple comparison blocks (e.g., 2025 vs 2024 and 2024 vs 2023).
The same year may appear more than once.

DUPLICATE RULE:
If the same (year, period_reported, metric_type, segment) appears more than once, keep only one copy.

────────────────────────────────────
VALUE RULES
────────────────────────────────────
• Extract only true Revenue or Transactions values  
• Ignore percentage change columns  
• Remove currency symbols and commas  
• Convert to numbers  
• Do not guess missing months or totals  
• Do not use notes or footnotes as data  

────────────────────────────────────
DATE RULES
────────────────────────────────────
• If the document contains “As of”, “As at”, “For the year ended”, or similar, extract it as as_at_date  
• Convert as_at_date to ISO format: YYYY-MM-DD  
• If no date is present, use null  

────────────────────────────────────
FIELD RULES
────────────────────────────────────
• metric_type must be either "Revenue" or "Transactions" only  
• asset_type, asset, and agency must be inferred from the document  
• source_url is supplied by the system — do not invent or modify it  
• If source_url is missing or not a valid URL,or use https://www.panynj.gov/content/dam/bridges-tunnels/pdfs/traffic-e-zpass-usage-2022.pdf


────────────────────────────────────
OUTPUT RULES (MANDATORY)
────────────────────────────────────
• Output ONLY a valid JSON array  
• No markdown  
• No explanation  
• No text before or after JSON  
• Never guess  
• If a field is unclear, use null  

Schema:
[
  {
    "asset_type": string | null,
    "asset": string | null,
    "agency": string | null,
    "as_at_date": string | null,
    "year": number | null,
    "period_reported": string | null,
    "metric_type": "Revenue" | "Transactions" | null,
    "segment": string | null,
    "value": number | null,
    "source": url | null,
    "source_url": url | null
  }
]

"""


# LLM CALL

def map_chunk_with_gemini(chunk: Dict) -> List[Dict]:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            SYSTEM_PROMPT,
            f"TEXT:\n{chunk['text']}"
        ]
    )

    return safe_json_parse(response.text)




# PIPELINE

def run_pipeline() -> List[Dict]:
    all_rows = []

    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        end_page = END_PAGE or total_pages

        print(f"PDF has {total_pages} pages")
        print(f"Processing pages {START_PAGE} → {end_page}")

        for page_no in range(START_PAGE, min(end_page, total_pages) + 1):
            page = pdf.pages[page_no - 1]

            tables = page.extract_tables()
            if tables:
                text = tables_to_text(tables)
            else:
                text = page.extract_text()
                if not text or len(text.strip()) < 50:
                    text = ocr_page(PDF_PATH, page_no)

            if not text:
                continue

            chunks = build_chunks(text, page_no)
            print(f"Page {page_no} → {len(chunks)} table chunks")

            for chunk in chunks:
                rows = map_chunk_with_gemini(chunk)
                print(f"  Chunk → {len(rows)} rows")

                for r in rows:
                    r.setdefault("asset_type", ASSET_TYPE)
                    r.setdefault("agency", AGENCY)
                    r.setdefault("source_url", SOURCE_URL)

                all_rows.extend(rows)

    return all_rows


# SAVE OUTPUT

def save_output(rows: List[Dict]):
    if not rows:
        print("No records extracted")
        return

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Extracted {len(df)} rows")
    print(f"Saved to {OUTPUT_CSV}")


# MAIN

if __name__ == "__main__":
    data = run_pipeline()
    save_output(data)
