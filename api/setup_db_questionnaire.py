import psycopg2
import os

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")

conn = psycopg2.connect(
    host=POSTGRES_HOST,
    port=5432,
    dbname="docscan",
    user="docscan",
    password="docscan",
)

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS questionnaire_extractions (
        id SERIAL PRIMARY KEY,
        classification_id INTEGER REFERENCES classifications(id),
        questionnaire_title TEXT,
        respondent_name TEXT,
        date_field TEXT,
        raw_ocr_text TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
""")

conn.commit()
cur.close()
conn.close()

print("Table 'questionnaire_extractions' ready.")
