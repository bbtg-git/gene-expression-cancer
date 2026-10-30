import streamlit as st
import boto3
from datetime import datetime

st.set_page_config(
    page_title="Cancer Gene Expression Analysis",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Gene Expression Data Analysis")
st.subheader("Cloud-Based Cancer Diagnosis Research Platform")

st.info(
    "Upload a gene expression CSV dataset. "
    "This starter version stores the uploaded dataset securely in AWS S3. "
    "It does not yet perform cancer classification."
)

uploaded_file = st.file_uploader(
    "Upload gene expression CSV",
    type=["csv"]
)

if uploaded_file is not None:
    st.write("**Uploaded file:**", uploaded_file.name)

    try:
        import pandas as pd

        df = pd.read_csv(uploaded_file)
        st.write("Dataset preview")
        st.dataframe(df.head(10))

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        if st.button("Upload dataset to AWS S3"):
            session = boto3.Session(
                aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                region_name="ap-south-1"
            )

            s3 = session.client("s3")

            uploaded_file.seek(0)

            object_key = (
                "input/gene_expression_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".csv"
            )

            s3.upload_fileobj(
                uploaded_file,
                "gene-expression-cancer-cc",
                object_key
            )

            st.success("Dataset uploaded successfully to AWS S3!")
            st.write("S3 object:", object_key)

    except Exception as error:
        st.error(f"Error: {error}")
