import boto3
import botocore


BUCKET_NAME = 'aslive-ml-models'
KEY = 'models/test_folder/test_file.txt'
ACCESS_ID = ''
ACCESS_KEY = ''

s3 = boto3.resource('s3',
                    aws_access_key_id=ACCESS_ID,
                    aws_secret_access_key=ACCESS_KEY)

try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, 'test_dir/mytest.txt')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise