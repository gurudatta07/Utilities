from google import colab
import tensorflow as tf
import os
import json

# Set up Google Cloud Storage for TPU.
# :param tpu_address: network address of the TPU, starting with 'grpc://'. Default: Colab's TPU address.
def setup_gcs(tpu_address: str = None):
    colab.auth.authenticate_user()

    tpu_address = tpu_address

    with tf.Session(tpu_address) as sess:
        with open('/content/adc.json', 'r') as f:
            auth_info = json.load(f)
        tf.contrib.cloud.configure_gcs(sess, credentials=auth_info)

# Upload one or more files from physical computer to Colab's virtual machine.
def upload_files():
    colab.files.upload()

# Download a file from Colab's virtual machine to physical computer.
# :param fn: file name on Colab
def download_file(fn: str):
    colab.files.download(fn)

# Mount Google Drive. Do nothing if Google Drive is already mounted.
# :param gdrive_path: local path to mount Google Drive to.
def mount_google_drive(gdrive_path: str = './gdrive'):
    colab.drive.mount(gdrive_path)
