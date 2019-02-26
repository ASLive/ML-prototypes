# TODO: update requirements.txt
import boto3
import botocore
import os


BUCKET_NAME = 'aslive-ml-models'
KEY = 'models/test_folder/'
# create ./aws_creds.txt file
# with aws s3 access_id in first line
# and access_key in second line
creds = [line.strip().decode("utf-8") for line in open("aws_creds.txt","rb")]
ACCESS_ID = creds[0];
ACCESS_KEY = creds[1];

s3 = boto3.resource('s3',
                    aws_access_key_id=ACCESS_ID,
                    aws_secret_access_key=ACCESS_KEY)

# try:
#     s3.Bucket(BUCKET_NAME).download_file(KEY,'.')
# except botocore.exceptions.ClientError as e:
#     if e.response['Error']['Code'] == "404":
#         print("The object does not exist.")
#     else:
#         raise

for s3_object in s3.Bucket(BUCKET_NAME).objects.all():
    # Need to split s3_object.key into path and file name, else it will give error file not found.
    path, filename = os.path.split(s3_object.key)

    print("path: " + path)
    print("filename: " + filename + "\n")
    #print(os.path.basename(path))
    if ('models/test_folder' == path) and (filename != ""):
        print("match\n")
        try:
            s3.Bucket(BUCKET_NAME).download_file(s3_object.key, os.path.basename(s3_object.key) + filename)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
