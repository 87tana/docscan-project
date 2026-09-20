import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

BUCKET_NAME = "docscan-uploads"

existing_buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]

if BUCKET_NAME not in existing_buckets:
    s3.create_bucket(Bucket=BUCKET_NAME)
    print(f"Created bucket: {BUCKET_NAME}")
else:
    print(f"Bucket already exists: {BUCKET_NAME}")
