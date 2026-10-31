import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="Gene Expression Cancer Analysis",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Gene Expression Data Analysis")
st.caption("Educational research demo — not for clinical diagnosis or medical decisions.")

st.write(
    "Upload a CSV with one sample per row, numeric gene-expression columns, "
    "and optionally a `label` column containing each sample's cancer type."
)

uploaded_file = st.file_uploader("Upload gene-expression CSV", type=["csv"])

if uploaded_file is None:
    st.info("Choose a CSV file to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read this CSV: {error}")
    st.stop()

if df.empty:
    st.error("The CSV has no data rows.")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Samples (rows)", f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

st.subheader("Data quality")
quality = pd.DataFrame({
    "column": df.columns,
    "data type": [str(dtype) for dtype in df.dtypes],
    "missing values": df.isna().sum().values,
    "unique values": [df[col].nunique(dropna=True) for col in df.columns],
})
st.dataframe(quality, use_container_width=True)

label_column = "label" if "label" in df.columns else None

if label_column:
    st.subheader("Cancer-type label distribution")
    label_counts = df[label_column].fillna("Missing label").value_counts()
    st.bar_chart(label_counts)
else:
    st.warning(
        "No column named `label` was found. Dataset summaries and expression "
        "visualizations are available, but supervised classification needs labels."
    )

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
if label_column in numeric_columns:
    numeric_columns.remove(label_column)

if not numeric_columns:
    st.warning("No numeric gene-expression columns were found in this CSV.")
    st.stop()

st.subheader("Gene expression overview")
selected_genes = st.multiselect(
    "Choose gene columns to visualize",
    options=numeric_columns,
    default=numeric_columns[: min(5, len(numeric_columns))],
)

if selected_genes:
    st.write("Mean expression by gene")
    st.bar_chart(df[selected_genes].mean())

    st.write("Expression values for the first 100 samples")
    st.line_chart(df[selected_genes].head(100))

if label_column:
    st.subheader("Train a basic cancer-type classifier")

    valid_rows = df[label_column].notna()
    labeled_df = df.loc[valid_rows].copy()
    X = labeled_df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    y = labeled_df[label_column].astype(str)

    class_counts = y.value_counts()
    st.write("Samples per class:")
    st.dataframe(class_counts.rename("samples"), use_container_width=True)

    if len(class_counts) < 2:
        st.warning("Classification requires at least two different label classes.")
    elif class_counts.min() < 2:
        st.warning(
            "Each label class needs at least two samples for a stratified train/test split."
        )
    else:
        test_fraction = st.slider(
            "Test data percentage",
            min_value=20,
            max_value=40,
            value=25,
            step=5,
        )

        if st.button("Train and evaluate model"):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_fraction / 100,
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

                accuracy = accuracy_score(y_test, predictions)
                st.metric("Test accuracy", f"{accuracy:.3f}")

                st.write("Classification report")
                report = classification_report(
                    y_test,
                    predictions,
                    output_dict=True,
                    zero_division=0,
                )
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

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
                st.error(f"Could not train the model: {error}")

st.divider()
st.subheader("Download cleaned data")
cleaned_csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=cleaned_csv,
    file_name="gene_expression_data.csv",
    mime="text/csv",
)
