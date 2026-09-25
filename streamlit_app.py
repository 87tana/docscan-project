from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import psycopg2
import pandas as pd
import requests

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")

def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=5432,
        dbname="docscan",
        user="docscan",
        password="docscan",
    )

st.set_page_config(page_title="DocScan", layout="wide")
st.title("DocScan — Document Digitization")

tab1, tab2 = st.tabs(["📤 Upload", "🔍 Browse & Search"])

# --- TAB 1: Upload ---
with tab1:
    st.subheader("Upload a new document")
    uploaded_file = st.file_uploader("Choose a document image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, width=250)
        with col2:
            if st.button("Classify & Extract", type="primary"):
                with st.spinner("Processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post("http://localhost:8000/predict", files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Predicted: **{result['predicted_class']}** ({result['confidence']:.2%} confidence)")
                    st.write("**Extracted fields:**")
                    st.json(result["extraction"])
                else:
                    st.error("Prediction failed.")

# --- TAB 2: Browse & Search ---
with tab2:
    st.subheader("Search documents")

    conn = get_db_connection()
    df = pd.read_sql(
        "SELECT id, filename, predicted_class, confidence, created_at FROM classifications ORDER BY created_at DESC;",
        conn
    )
    conn.close()

    col1, col2 = st.columns(2)
    with col1:
        class_filter = st.selectbox("Filter by type:", ["All"] + sorted(df["predicted_class"].unique().tolist()))
    with col2:
        name_filter = st.text_input("Search by filename:")

    filtered_df = df.copy()
    if class_filter != "All":
        filtered_df = filtered_df[filtered_df["predicted_class"] == class_filter]
    if name_filter:
        filtered_df = filtered_df[filtered_df["filename"].str.contains(name_filter, case=False)]

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("View document details")

    if len(filtered_df) > 0:
        selected_id = st.selectbox("Select a document ID to inspect:", filtered_df["id"])

        if selected_id:
            conn = get_db_connection()
            row = pd.read_sql(
                f"SELECT filename, predicted_class, confidence, created_at FROM classifications WHERE id = {selected_id};",
                conn
            )
            predicted_class = row["predicted_class"].iloc[0]

            st.write("**Classification:**")
            st.dataframe(row, use_container_width=True, hide_index=True)

            extraction_table = f"{predicted_class}_extractions"
            extraction_df = pd.read_sql(
                f"SELECT * FROM {extraction_table} WHERE classification_id = {selected_id};", conn
            )
            conn.close()

            st.write(f"**Extracted fields ({predicted_class}):**")

            if len(extraction_df) > 0:
                editable_cols = [c for c in extraction_df.columns if c not in ("id", "classification_id", "raw_ocr_text", "created_at")]

                with st.form(key=f"edit_form_{selected_id}"):
                    new_values = {}
                    for col in editable_cols:
                        current_value = extraction_df[col].iloc[0]
                        new_values[col] = st.text_input(col, value="" if pd.isna(current_value) else str(current_value))

                    submitted = st.form_submit_button("Save changes")

                    if submitted:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        set_clause = ", ".join([f"{col} = %s" for col in editable_cols])
                        values = list(new_values.values()) + [selected_id]
                        cur.execute(
                            f"UPDATE {extraction_table} SET {set_clause} WHERE classification_id = %s;",
                            values
                        )
                        conn.commit()
                        cur.close()
                        conn.close()
                        st.success("Saved! Refresh to see updated values.")
    else:
        st.info("No documents match your filters.")
