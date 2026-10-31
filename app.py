import io
from datetime import datetime, timezone
import json
from datetime import datetime, timezone
from io import BytesIO
import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# ---------- Page setup ----------
st.set_page_config(
    page_title="Cancer Gene Expression Analysis",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Gene Expression Data Analysis")
st.caption(
    "Educational research demo only. Not for clinical diagnosis or medical decisions."
)

st.write(
    "Upload a CSV with one sample per row, numeric gene-expression columns, "
    "and optionally a `label` column containing the sample's cancer type."
)

# ---------- S3 upload ----------
def save_uploaded_csv_to_s3(uploaded_file):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name="ap-south-1",
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    object_key = f"input/{timestamp}_{safe_name}"

    uploaded_file.seek(0)
    s3.upload_fileobj(
        uploaded_file,
        "gene-expression-cancer-cc",
        object_key,
        ExtraArgs={"ContentType": "text/csv"},
    )

    return object_key


# ---------- Upload and read CSV ----------
uploaded_file = st.file_uploader(
    "Upload gene-expression CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Choose a CSV file to begin.")
    st.stop()

try:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

if df.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

# ---------- Dataset overview ----------
st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Samples (rows)", f"{df.shape[0]:,}")
col2.metric("Columns", f"{df.shape[1]:,}")
col3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

st.subheader("Data quality")
quality_df = pd.DataFrame({
    "Column": df.columns,
    "Data type": [str(dtype) for dtype in df.dtypes],
    "Missing values": df.isna().sum().values,
    "Unique values": [df[column].nunique(dropna=True) for column in df.columns],
})
st.dataframe(quality_df, use_container_width=True)

# ---------- S3 button ----------
st.subheader("Save dataset to AWS S3")
st.write("This uploads the original CSV to your configured private S3 bucket.")

if st.button("Upload dataset to S3"):
    try:
        s3_key = save_uploaded_csv_to_s3(uploaded_file)
        st.success("Dataset uploaded successfully.")
        st.code(f"s3://gene-expression-cancer-cc/{s3_key}")
    except KeyError:
        st.error(
            "AWS credentials are missing from Streamlit Secrets. "
            "Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your app settings."
        )
    except Exception as error:
        st.error(f"S3 upload failed: {error}")

# ---------- Expression analysis ----------
label_column = "label" if "label" in df.columns else None

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
if label_column in numeric_columns:
    numeric_columns.remove(label_column)

st.subheader("Gene expression overview")

if not numeric_columns:
    st.warning("No numeric gene-expression columns were found.")
else:
    selected_genes = st.multiselect(
        "Select gene columns to visualize",
        options=numeric_columns,
        default=numeric_columns[: min(5, len(numeric_columns))],
    )

    if selected_genes:
        st.write("Mean expression by gene")
        st.bar_chart(df[selected_genes].mean())

        st.write("Expression values for the first 100 samples")
        st.line_chart(df[selected_genes].head(100))

# ---------- Label distribution ----------
if label_column:
    st.subheader("Cancer-type label distribution")
    label_counts = df[label_column].fillna("Missing label").value_counts()
    st.bar_chart(label_counts)
else:
    st.info(
        "No `label` column was found. You can explore the data, "
        "but supervised classification requires known labels."
    )
def upload_analysis_report_to_s3(report_df, report_name="gene_expression_analysis"):
    """Upload an analysis report DataFrame to the S3 output/ folder."""
    try:
        s3 = boto3.client(
            "s3",
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "ap-south-1"),
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        key = f"output/{report_name}_{timestamp}.csv"

        csv_bytes = report_df.to_csv(index=False).encode("utf-8")
        s3.upload_fileobj(
            BytesIO(csv_bytes),
            "gene-expression-cancer-cc",
            key,
            ExtraArgs={"ContentType": "text/csv"},
        )

        return f"s3://gene-expression-cancer-cc/{key}"

    except Exception as exc:
        st.error(f"S3 upload failed: {exc}")
        return None

if st.button("Upload analysis output to S3"):
    # Summary of the uploaded dataset
    summary_rows = [
        {"metric": "sample_count", "value": len(df)},
        {"metric": "column_count", "value": len(df.columns)},
        {"metric": "missing_values", "value": int(df.isna().sum().sum())},
    ]

    # Include the cancer-type distribution, if the column exists
    if "cancer_type" in df.columns:
        counts = df["cancer_type"].value_counts()
        for cancer_type, count in counts.items():
            summary_rows.append({
                "metric": f"count_{cancer_type}",
                "value": int(count),
            })

    report_df = pd.DataFrame(summary_rows)
    s3_uri = upload_analysis_report_to_s3(report_df)

    if s3_uri:
        st.success(f"Analysis report uploaded: {s3_uri}")
        st.download_button(
            "Download analysis report",
            data=report_df.to_csv(index=False),
            file_name="gene_expression_analysis_report.csv",
            mime="text/csv",
        )

# ---------- Classifier ----------
if label_column and numeric_columns:
    st.subheader("Basic cancer-type classification")

    labeled_df = df.loc[df[label_column].notna()].copy()
    X = labeled_df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    y = labeled_df[label_column].astype(str)

    class_counts = y.value_counts()
    st.write("Samples per label:")
    st.dataframe(class_counts.rename("Samples"), use_container_width=True)

    if len(class_counts) < 2:
        st.warning("Classification requires at least two label classes.")
    elif class_counts.min() < 2:
        st.warning("Each class needs at least two samples for a stratified split.")
    else:
        test_percent = st.slider(
            "Test set size (%)",
            min_value=20,
            max_value=40,
            value=25,
            step=5,
        )

        if st.button("Train and evaluate classifier"):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_percent / 100,
                    random_state=42,
                    stratify=y,
                )

                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced",
                    ),
                )

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                st.metric(
                    "Test accuracy",
                    f"{accuracy_score(y_test, predictions):.3f}",
                )

                st.write("Classification report")
                report = classification_report(
                    y_test,
                    predictions,
                    output_dict=True,
                    zero_division=0,
                )
                st.dataframe(
                    pd.DataFrame(report).transpose(),
                    use_container_width=True,
                )

                st.write("Confusion matrix")
                labels = sorted(y.unique())
                matrix = confusion_matrix(y_test, predictions, labels=labels)
                st.dataframe(
                    pd.DataFrame(
                        matrix,
                        index=[f"Actual: {label}" for label in labels],
                        columns=[f"Predicted: {label}" for label in labels],
                    ),
                    use_container_width=True,
                )

            except Exception as error:
                st.error(f"Model training failed: {error}")

# ---------- Download ----------
st.subheader("Download dataset")
csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    data=csv_data,
    file_name="gene_expression_data.csv",
    mime="text/csv",
)
