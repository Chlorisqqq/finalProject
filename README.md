# Customer Email Triage Dashboard

Streamlit app for uploading customer support emails from Excel, classifying issue categories with a fine-tuned text classification model, reviewing ticket details, viewing issue distribution, and exporting processed results.

## Expected Excel Columns

The uploaded workbook must include:

- `subject`
- `email_body`

The app automatically generates `email_id` values during processing, so the uploaded file does not need an ID column.

## Configure The Model

The app uses this Hugging Face classification model by default:

```text
ysubr/CustomModel_queue_bert
```

You can still override it with `CLASSIFICATION_MODEL_PATH` if you want to use another local fine-tuned model directory or Hugging Face model name.

PowerShell example:

```powershell
$env:CLASSIFICATION_MODEL_PATH="ysubr/CustomModel_queue_bert"
streamlit run app.py
```

The summarization function is intentionally a placeholder and currently returns:

```text
Summary model not implemented yet.
```

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```
