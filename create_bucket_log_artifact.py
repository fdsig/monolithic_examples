import os
import boto3
import requests
import gzip
import shutil
import wandb

# Set AWS authentication
os.environ['AWS_ACCESS_KEY_ID'] = "***"
os.environ['AWS_SECRET_ACCESS_KEY'] = "***"
os.environ['AWS_SESSION_TOKEN'] = "***"
# Define the bucket name and region
BUCKET_NAME = "hans-gong-test-bucket-3"
REGION = "us-west-2"  # or any other region
wandb.login(host='https://gong.sandbox-aws.wandb.ml/',key='local-***')
# Create an S3 client
s3 = boto3.client('s3', region_name=REGION)

# Create the bucket
s3.create_bucket(
    Bucket=BUCKET_NAME,
    CreateBucketConfiguration={'LocationConstraint': REGION}
)

# URLs for the MNIST data
urls = [
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
]

for url in urls:
    # Download the data
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    # Decompress the data
    with gzip.open(filename, 'rb') as f_in:
        with open(filename[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Define target path
    target_path = f"mnist/{'train' if 'train' in filename else 'test'}/{filename[:-3]}"

    # Upload the data to the bucket
    s3.upload_file(filename[:-3], BUCKET_NAME, target_path)

print(f"Bucket {BUCKET_NAME} created and MNIST data uploaded.")



# Initialize wandb run
run = wandb.init(project='gong_artifact', entity='testing-artifacts-team')

# Create an artifact
artifact = wandb.Artifact(name='background',type='from_s3')

# Add reference to the artifact
artifact.add_reference(f's3://{BUCKET_NAME}/mnist')

# Log the artifact
run.log_artifact(artifact)

run.finish()
