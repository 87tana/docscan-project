import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="docscan",
    user="docscan",
    password="docscan",
)

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS classifications (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        storage_path TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence FLOAT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
""")

conn.commit()
cur.close()
conn.close()

print("Table 'classifications' ready.")
