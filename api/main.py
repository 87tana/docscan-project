from dotenv import load_dotenv
load_dotenv()

import os
import logging
import mlflow
import mlflow.pytorch
from fastapi import FastAPI

import io
import torch
from PIL import Image
from fastapi import File, UploadFile
from torchvision import transforms

import boto3
import psycopg2
from datetime import datetime
import uuid

import easyocr
import anthropic
import json

from transformers import RobertaTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

from api.constants import CLASS_NAMES

logger = logging.getLogger("docscan")
logger.setLevel(logging.INFO)
logger.handlers = []  # clear any existing handlers first, no matter how many
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.propagate = False

# --- MLflow setup: use hosted registry (RDS+S3) if configured, else fall back to local ---
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/home/tannaz/Documents/Projects/Docscan_project")

RDS_HOST = os.environ.get("RDS_HOST")

if RDS_HOST:
    RDS_PORT = os.environ["RDS_PORT"]
    RDS_USER = os.environ["RDS_USER"]
    RDS_PASSWORD = os.environ["RDS_PASSWORD"]
    RDS_DB_NAME = os.environ["RDS_DB_NAME"]
    tracking_uri = f"postgresql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"
    mlflow.set_tracking_uri(tracking_uri)
else:
    MLFLOW_LOCAL_DIR = os.path.join(PROJECT_ROOT, "mlflow_local")
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(MLFLOW_LOCAL_DIR, 'mlruns.db')}")

MODEL_NAME = "docscan-classifier"
MODEL_STAGE_OR_VERSION = "1"

app = FastAPI(title="DocScan Classifier API")


#prometheus

PREDICTION_COUNTER = Counter(
    "docscan_predictions_total", "Total predictions made", ["predicted_class"]
)
PREDICTION_CONFIDENCE = Histogram(
    "docscan_prediction_confidence", "Confidence score of predictions", ["predicted_class"]
)
EXTRACTION_ERRORS = Counter(
    "docscan_extraction_errors_total", "Total extraction errors", ["predicted_class"]
)



model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE_OR_VERSION}")

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_STAGE_OR_VERSION}


@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/stats")
def stats():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT predicted_class, COUNT(*), AVG(confidence) FROM classifications GROUP BY predicted_class;")
    class_stats = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM classifications;")
    total = cur.fetchone()[0]

    cur.close()
    conn.close()

    return {
        "total_predictions": total,
        "by_class": [
            {"class": row[0], "count": row[1], "avg_confidence": round(row[2], 4)}
            for row in class_stats
        ],
    }


IMAGE_SIZE = 224
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

#CLASS_NAMES = ["form", "invoice", "handwritten", "questionnaire", "fallback"]

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_value = round(confidence.item(), 4)

    logger.info(f"Prediction: class={predicted_class} confidence={confidence_value} filename={file.filename}")
    PREDICTION_COUNTER.labels(predicted_class=predicted_class).inc()
    PREDICTION_CONFIDENCE.labels(predicted_class=predicted_class).observe(confidence_value)
    #print(f"PRINT TEST: class={predicted_class}")

    # --- Save file to MinIO ---
    file_id = str(uuid.uuid4())
    storage_key = f"{file_id}_{file.filename}"
    s3.put_object(Bucket=BUCKET_NAME, Key=storage_key, Body=image_bytes)

    # --- Save result to Postgres ---
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO classifications (filename, storage_path, predicted_class, confidence)
        VALUES (%s, %s, %s, %s)
        """,
        (file.filename, storage_key, predicted_class, confidence_value),
    )
    conn.commit()
    cur.close()
    conn.close()
    extraction_result = None

    if predicted_class == "invoice":
        try:
            extracted_fields = extract_invoice_fields(image)

            conn2 = get_db_connection()
            cur2 = conn2.cursor()
            cur2.execute(
                """
                INSERT INTO invoice_extractions
                (classification_id, vendor_name, invoice_date, invoice_number, total_amount, raw_ocr_text)
                VALUES (
                    (SELECT id FROM classifications WHERE storage_path = %s),
                    %s, %s, %s, %s, %s
                )
                """,
                (
                    storage_key,
                    extracted_fields.get("vendor_name"),
                    extracted_fields.get("invoice_date"),
                    extracted_fields.get("invoice_number"),
                    extracted_fields.get("total_amount"),
                    extracted_fields.get("raw_ocr_text"),
                ),
            )
            conn2.commit()
            cur2.close()
            conn2.close()

            extraction_result = extracted_fields
        except Exception as e:
            logger.error(f"Extraction failed for invoice (storage_key={storage_key}): {e}")
            EXTRACTION_ERRORS.labels(predicted_class=predicted_class).inc()
            extraction_result = {"error": str(e)}

    elif predicted_class == "fallback":
        try:
            extracted_fields = extract_fallback_text(image)

            conn2 = get_db_connection()
            cur2 = conn2.cursor()
            cur2.execute(
                """
                INSERT INTO fallback_extractions
                (classification_id, raw_ocr_text)
                VALUES (
                    (SELECT id FROM classifications WHERE storage_path = %s),
                    %s
                )
                """,
                (storage_key, extracted_fields.get("raw_ocr_text")),
            )
            conn2.commit()
            cur2.close()
            conn2.close()

            extraction_result = extracted_fields
        except Exception as e:
            logger.error(f"Extraction failed for fallback (storage_key={storage_key}): {e}")
            EXTRACTION_ERRORS.labels(predicted_class=predicted_class).inc()            
            extraction_result = {"error": str(e)}

    elif predicted_class == "form":
        try:
            extracted_fields = extract_form_fields(image)

            conn2 = get_db_connection()
            cur2 = conn2.cursor()
            cur2.execute(
                """
                INSERT INTO form_extractions
                (classification_id, form_title, person_name, date_field, raw_ocr_text)
                VALUES (
                    (SELECT id FROM classifications WHERE storage_path = %s),
                    %s, %s, %s, %s
                )
                """,
                (
                    storage_key,
                    extracted_fields.get("form_title"),
                    extracted_fields.get("person_name"),
                    extracted_fields.get("date_field"),
                    extracted_fields.get("raw_ocr_text"),
                ),
            )
            conn2.commit()
            cur2.close()
            conn2.close()

            extraction_result = extracted_fields
        except Exception as e:
            logger.error(f"Extraction failed for form (storage_key={storage_key}): {e}")
            EXTRACTION_ERRORS.labels(predicted_class=predicted_class).inc()
            extraction_result = {"error": str(e)}

    elif predicted_class == "questionnaire":
        try:
            extracted_fields = extract_questionnaire_fields(image)

            conn2 = get_db_connection()
            cur2 = conn2.cursor()
            cur2.execute(
                """
                INSERT INTO questionnaire_extractions
                (classification_id, questionnaire_title, respondent_name, date_field, raw_ocr_text)
                VALUES (
                    (SELECT id FROM classifications WHERE storage_path = %s),
                    %s, %s, %s, %s
                )
                """,
                (
                    storage_key,
                    extracted_fields.get("questionnaire_title"),
                    extracted_fields.get("respondent_name"),
                    extracted_fields.get("date_field"),
                    extracted_fields.get("raw_ocr_text"),
                ),
            )
            conn2.commit()
            cur2.close()
            conn2.close()

            extraction_result = extracted_fields
        except Exception as e:
            logger.error(f"Extraction failed for questionnaire (storage_key={storage_key}): {e}")
            EXTRACTION_ERRORS.labels(predicted_class=predicted_class).inc()
            extraction_result = {"error": str(e)}

    elif predicted_class == "handwritten":
        try:
            extracted_fields = extract_handwritten_fields(image)

            conn2 = get_db_connection()
            cur2 = conn2.cursor()
            cur2.execute(
                """
                INSERT INTO handwritten_extractions
                (classification_id, summary, raw_ocr_text)
                VALUES (
                    (SELECT id FROM classifications WHERE storage_path = %s),
                    %s, %s
                )
                """,
                (storage_key, extracted_fields.get("summary"), extracted_fields.get("raw_ocr_text")),
            )
            conn2.commit()
            cur2.close()
            conn2.close()

            extraction_result = extracted_fields
        except Exception as e:
            logger.error(f"Extraction failed for handwritten (storage_key={storage_key}): {e}")
            EXTRACTION_ERRORS.labels(predicted_class=predicted_class).inc()
            extraction_result = {"error": str(e)}

    return {
        "predicted_class": predicted_class,
        "confidence": confidence_value,
        "storage_path": storage_key,
        "extraction": extraction_result,
    }

# --- Storage clients (created once at startup) ---
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)
BUCKET_NAME = "docscan-uploads"


POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")

def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=5432,
        dbname="docscan",
        user="docscan",
        password="docscan",
    )

# --- OCR + LLM setup (created once at startup) ---
ocr_reader = easyocr.Reader(['en'])
llm_client = anthropic.Anthropic()

trocr_tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
trocr_feature_extractor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def extract_invoice_fields(image: Image.Image) -> dict:
    import numpy as np
    ocr_results = ocr_reader.readtext(np.array(image))
    raw_text = " ".join([text for (bbox, text, confidence) in ocr_results])

    prompt = f"""You are extracting structured data from noisy OCR text of a scanned invoice.
The text may contain OCR errors - use context to infer correct values where possible.

Extract these fields and return ONLY valid JSON, no other text:
- vendor_name
- invoice_date
- invoice_number
- total_amount (as a number, or null)

OCR text:
{raw_text}
"""

    response = llm_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    result_text = response.content[0].text
    result_text = result_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    fields = json.loads(result_text)
    fields["raw_ocr_text"] = raw_text
    return fields


def extract_fallback_text(image: Image.Image) -> dict:
    import numpy as np
    ocr_results = ocr_reader.readtext(np.array(image))
    raw_text = " ".join([text for (bbox, text, confidence) in ocr_results])
    return {"raw_ocr_text": raw_text}


def extract_form_fields(image: Image.Image) -> dict:
    import numpy as np
    ocr_results = ocr_reader.readtext(np.array(image))
    raw_text = " ".join([text for (bbox, text, confidence) in ocr_results])

    prompt = f"""You are extracting structured data from noisy OCR text of a scanned form document.
The text may contain OCR errors - use context to infer correct values where possible.

Extract these fields and return ONLY valid JSON, no other text:
- form_title (the form's name/type, or null)
- person_name (any person's name on the form, or null)
- date_field (any date present, or null)

OCR text:
{raw_text}
"""

    response = llm_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    result_text = response.content[0].text
    result_text = result_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    fields = json.loads(result_text)
    fields["raw_ocr_text"] = raw_text
    return fields


def extract_questionnaire_fields(image: Image.Image) -> dict:
    import numpy as np
    ocr_results = ocr_reader.readtext(np.array(image))
    raw_text = " ".join([text for (bbox, text, confidence) in ocr_results])

    prompt = f"""You are extracting structured data from noisy OCR text of a scanned questionnaire.
The text may contain OCR errors - use context to infer correct values where possible.

Extract these fields and return ONLY valid JSON, no other text:
- questionnaire_title (the questionnaire's name/topic, or null)
- respondent_name (the person who filled it out, if present, or null)
- date_field (any date present, or null)

OCR text:
{raw_text}
"""

    response = llm_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    result_text = response.content[0].text
    result_text = result_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    fields = json.loads(result_text)
    fields["raw_ocr_text"] = raw_text
    return fields


def extract_handwritten_fields(image: Image.Image) -> dict:
    import numpy as np
    ocr_results = ocr_reader.readtext(np.array(image))

    line_texts = []
    for (bbox, text, confidence) in ocr_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)

        line_crop = image.crop((left, top, right, bottom))
        pixel_values = trocr_feature_extractor(images=line_crop, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=50)
        trocr_text = trocr_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        line_texts.append(trocr_text)

    raw_text = " ".join(line_texts)

    prompt = f"""This is noisy OCR text from a handwritten document, read line by line, so some words may be misspelled or garbled.

Write a short 1-2 sentence summary of what this document says.

OCR text:
{raw_text}
"""

    response = llm_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )

    summary = response.content[0].text.strip()

    return {"summary": summary, "raw_ocr_text": raw_text}
