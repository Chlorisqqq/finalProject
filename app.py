import os
from io import BytesIO

import pandas as pd
import streamlit as st
from transformers import pipeline

# =========================
# Config
# =========================

DEFAULT_CLASSIFICATION_MODEL_PATH = "ysubr/CustomModel_queue_bert"


def get_configured_model_path():
    """
    Read the model path from the environment or Streamlit secrets.
    """
    env_model_path = os.getenv("CLASSIFICATION_MODEL_PATH")
    if env_model_path:
        return env_model_path

    try:
        return st.secrets.get(
            "CLASSIFICATION_MODEL_PATH",
            DEFAULT_CLASSIFICATION_MODEL_PATH,
        )
    except Exception:
        return DEFAULT_CLASSIFICATION_MODEL_PATH


CLASSIFICATION_MODEL_PATH = get_configured_model_path()

REQUIRED_COLUMNS = ["subject", "email_body"]
OUTPUT_COLUMNS = [
    "email_id",
    "predicted_issue",
    "confidence",
    "summary",
    "subject",
    "email_body",
]


# =========================
# Model Loading Functions
# =========================

@st.cache_resource
def load_classification_model(model_path):
    """
    Load the fine-tuned text classification model.
    This model should classify each email into an issue category.
    """
    if not model_path:
        raise ValueError(
            "Please set CLASSIFICATION_MODEL_PATH to your fine-tuned model path "
            "before running AI analysis."
        )

    try:
        from transformers import pipeline
    except ImportError as error:
        raise ImportError(
            "Could not import transformers.pipeline. Please install or repair the "
            "Transformers package with `pip install -U transformers`."
        ) from error

    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
    )
    return classifier


@st.cache_resource
def load_summarization_model():
    """
    Load T5 model and tokenizer for summarization.
    """
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    return tokenizer, model

# =========================
# Data Processing Functions
# =========================

def read_excel_file(uploaded_file):
    """
    Read uploaded Excel file into a pandas DataFrame.
    """
    df = pd.read_excel(uploaded_file)
    return df


def validate_input_columns(df):
    """
    Check whether the uploaded Excel file contains required columns.
    Required columns:
    - subject
    - email_body
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Please upload an Excel file with columns: {REQUIRED_COLUMNS}"
        )


def prepare_email_text(row):
    """
    Combine subject and email body into one text input for classification.
    """
    subject = str(row["subject"]) if pd.notna(row["subject"]) else ""
    body = str(row["email_body"]) if pd.notna(row["email_body"]) else ""

    email_text = f"Subject: {subject}\n\nEmail Body: {body}"
    return email_text


# =========================
# AI Pipeline Functions
# =========================

def classify_email(email_text, classifier):
    """
    Classify one email and return predicted issue category and confidence score.
    """
    result = classifier(
        email_text,
        truncation=True,
        max_length=512,
    )

    prediction = result[0]

    issue_category = prediction["label"]
    confidence = prediction["score"]

    return issue_category, confidence


def summarize_email(email_text, summarizer=None):
    """
    Generate a summary using T5 model.
    """
    if summarizer is None:
        return "Summarization model is not loaded."

    tokenizer, model = summarizer

    try:
        prompt = "summarize: " + email_text

        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        summary_ids = model.generate(
            inputs,
            max_length=60,
            min_length=15,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as error:
        return f"Summary generation failed: {error}"


def analyze_emails(df, classifier, summarizer=None):
    """
    Run the full pipeline:
    1. Combine subject and email body
    2. Classify email issue category
    3. Generate email summary placeholder
    4. Return processed DataFrame
    """
    processed_rows = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(df)

    for index, row in df.iterrows():
        email_text = prepare_email_text(row)

        issue_category, confidence = classify_email(email_text, classifier)
        summary = summarize_email(email_text, summarizer)

        processed_rows.append(
            {
                "email_id": index + 1,
                "subject": row["subject"],
                "email_body": row["email_body"],
                "predicted_issue": issue_category,
                "confidence": confidence,
                "summary": summary,
            }
        )

        completed = len(processed_rows)
        progress_bar.progress(completed / total_rows)
        status_text.caption(f"Processed {completed} of {total_rows} emails")

    status_text.empty()
    progress_bar.empty()

    result_df = pd.DataFrame(processed_rows, columns=OUTPUT_COLUMNS)
    return result_df


# =========================
# Export Function
# =========================

def convert_df_to_excel(df):
    """
    Convert processed DataFrame into downloadable Excel file.
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Processed Emails")

    processed_data = output.getvalue()
    return processed_data


# =========================
# UI Helpers
# =========================

def get_current_file_key(uploaded_file):
    """
    Build a stable key so processed results reset when the user uploads a new file.
    """
    return f"{uploaded_file.name}-{uploaded_file.size}"


def show_sidebar():
    """
    Render sidebar navigation.
    """
    st.sidebar.title("Navigation")
    selected_section = st.sidebar.radio(
        "Go to",
        [
            "Upload Emails",
            "Dashboard",
            "Review Tickets",
            "Export Results",
        ],
    )
    return selected_section


def show_upload_page(raw_df, processed_df):
    """
    Show uploaded and processed data on the upload page.
    """
    st.subheader("Uploaded Email Data Preview")
    st.dataframe(raw_df.head(), use_container_width=True)

    if processed_df is not None:
        st.subheader("Processed Email Results")

        preview_columns = [
            "email_id",
            "predicted_issue",
            "confidence",
            "summary",
            "subject",
        ]

        st.dataframe(
            processed_df[preview_columns],
            use_container_width=True,
        )

def show_dashboard_page(processed_df):
    """
    Show issue distribution chart and table.
    """
    st.subheader("Issue Distribution Chart")

    issue_counts = processed_df["predicted_issue"].value_counts().reset_index()
    issue_counts.columns = ["Issue Category", "Email Count"]

    st.bar_chart(
        data=issue_counts,
        x="Issue Category",
        y="Email Count",
    )

    st.subheader("Issue Distribution Table")
    st.dataframe(issue_counts, use_container_width=True)


def show_review_tickets_page(processed_df):
    """
    Show processed ticket details in expanders.
    """
    st.subheader("Review Email Details")

    selected_issue = st.selectbox(
        "Filter by issue category",
        options=["All"] + sorted(processed_df["predicted_issue"].unique().tolist()),
    )

    if selected_issue == "All":
        filtered_df = processed_df
    else:
        filtered_df = processed_df[processed_df["predicted_issue"] == selected_issue]

    st.write(f"Showing {len(filtered_df)} emails.")

    for _, row in filtered_df.iterrows():
        expander_title = (
            f"Email ID: {row['email_id']} | "
            f"Issue: {row['predicted_issue']} | "
            f"Confidence: {row['confidence']:.2%}"
        )

        with st.expander(expander_title):
            st.markdown("**Subject:**")
            st.write(row["subject"])

            st.markdown("**Original Email:**")
            st.write(row["email_body"])

            st.markdown("**Predicted Issue Category:**")
            st.write(row["predicted_issue"])

            st.markdown("**Confidence Score:**")
            st.write(f"{row['confidence']:.2%}")

            st.markdown("**AI Summary:**")
            st.write(row["summary"])


def show_export_results_page(processed_df):
    """
    Allow users to download processed results as an Excel file.
    """
    st.subheader("Download Processed Results")

    excel_data = convert_df_to_excel(processed_df)

    st.download_button(
        label="Download Results as Excel",
        data=excel_data,
        file_name="processed_email_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =========================
# Main Streamlit App
# =========================

def main():
    st.set_page_config(
        page_title="Customer Email Triage Dashboard",
        layout="wide",
    )

    st.title("Customer Email Triage Dashboard")
    st.write(
        "Upload customer support emails, classify issue categories, "
        "generate summaries, and review the results in a business dashboard."
    )

    selected_section = show_sidebar()
    summarizer = load_summarization_model()

    uploaded_file = st.file_uploader(
        "Upload an Excel file",
        type=["xlsx", "xls"],
        help="The file should contain two columns: subject, email_body.",
    )

    if uploaded_file is None:
        st.info("Please upload an Excel file to begin.")
        return

    try:
        raw_df = read_excel_file(uploaded_file)
        validate_input_columns(raw_df)
    except Exception as error:
        st.error(f"File validation failed: {error}")
        return

    current_file_key = get_current_file_key(uploaded_file)

    if st.session_state.get("file_key") != current_file_key:
        st.session_state["file_key"] = current_file_key
        st.session_state.pop("processed_df", None)

    processed_df = st.session_state.get("processed_df")

    show_upload_page(raw_df, processed_df)

    if st.button("Run AI Analysis", type="primary"):
        try:
            classifier = load_classification_model(CLASSIFICATION_MODEL_PATH)

            with st.spinner("Classifying emails and generating summaries..."):
                processed_df = analyze_emails(
                    raw_df,
                    classifier=classifier,
                    summarizer=summarizer,
                )

            st.session_state["processed_df"] = processed_df
            st.success("Email analysis completed.")
            st.rerun()

        except Exception as error:
            st.error(f"AI analysis failed: {error}")
            return

    processed_df = st.session_state.get("processed_df")

    if processed_df is None:
        st.warning("Please click 'Run AI Analysis' to process the uploaded emails.")
        return

    if selected_section == "Dashboard":
        show_dashboard_page(processed_df)
    elif selected_section == "Review Tickets":
        show_review_tickets_page(processed_df)
    elif selected_section == "Export Results":
        show_export_results_page(processed_df)
