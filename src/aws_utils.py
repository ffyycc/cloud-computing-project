import os
import logging
from pathlib import Path
import boto3


logger = logging.getLogger(__name__)


def upload_artifacts(artifacts: Path, config: dict) -> list[str]:
    """Upload all the artifacts in the specified directory to S3

    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        config: Config required to upload artifacts to S3; see example config file for structure

    Returns:
        List of S3 uri's for each file that was uploaded
    """
    # Retrieve the S3 configuration from the config dictionary
    upload = config["upload"]
    bucket_name = config["bucket_name"]
    prefix = config["prefix"]

    # If the upload flag is set to False, skip the upload process
    if not upload:
        logger.info("Upload is disabled in the configuration.")
        return []

    # Create a boto3 session and S3 client
    session = boto3.Session()
    s3 = session.client("s3")

    # Initialize a list to store the S3 URIs of the uploaded files
    uploaded_uris = []

    try:
        # Iterate through all files in the artifacts directory
        for root, _, files in os.walk(artifacts):
            for file in files:
                file_path = os.path.join(root, file)
                # Create the S3 key by replacing the local artifacts path with the prefix
                s3_key = prefix + file_path.replace(str(artifacts), "", 1).lstrip(
                    os.sep
                )

                # Upload the file to S3
                s3.upload_file(Filename=file_path, Bucket=bucket_name, Key=s3_key)

                # Add the uploaded file's S3 URI to the list
                uploaded_uri = "s3://%s/%s" % (bucket_name, s3_key)
                uploaded_uris.append(uploaded_uri)

                logger.debug("Successfully uploaded %s to %s", file_path, uploaded_uri)

    except Exception as e:
        logger.error("Failed to upload files: %s", str(e))
        return []

    logger.info("All files have been uploaded successfully.")

    return uploaded_uris
