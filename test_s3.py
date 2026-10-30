import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = "gene-expression-cancer-cc"
REGION = "ap-south-1"

session = boto3.Session(
    profile_name="cloud-project",
    region_name=REGION
)

s3 = session.client("s3")

try:
    s3.head_bucket(Bucket=BUCKET_NAME)
    print("SUCCESS: Biology S3 connection established!")

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="input/test.txt",
        Body=b"Biology Cancer Gene Expression project connected!"
    )

    print("SUCCESS: Test file uploaded!")

    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix="input/"
    )

    for obj in response.get("Contents", []):
        print("Uploaded file:", obj["Key"])

except ClientError as error:
    print("AWS ERROR:", error)
