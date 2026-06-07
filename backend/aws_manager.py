# aws_manager.py
"""
AWS Cloud Integration for Dew-FDL

Handles connecting the Dew/Edge layer to real AWS Cloud Storage.
- Amazon DynamoDB: Storing metadata (labels, confidence, timestamps)
- Amazon S3: Storing actual disaster images (.jpg)

Dependencies:
    pip install boto3
"""

import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import uuid
import time
import os

# -------------------------------------------------------------------------
# AWS CONFIGURATION (Change these if your AWS setup is different)
# -------------------------------------------------------------------------
AWS_REGION = "us-east-1"
DYNAMODB_TABLE_NAME = "DewCloudAlerts"
S3_BUCKET_NAME = "dew-disaster-images"


def _get_aws_clients():
    """Returns (dynamodb_client, s3_client) if authenticated, else (None, None)."""
    try:
        sts = boto3.client("sts", region_name=AWS_REGION)
        sts.get_caller_identity() # Test authentication
        
        dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)
        s3 = boto3.client("s3", region_name=AWS_REGION)
        return dynamodb, s3
    except (NoCredentialsError, ClientError) as e:
        return None, None


def is_aws_configured() -> bool:
    """Check if AWS credentials are valid."""
    dynamo, s3 = _get_aws_clients()
    return dynamo is not None


def ensure_aws_infrastructure(dynamodb, s3):
    """Automatically create the DynamoDB table and S3 Bucket if they don't exist."""
    # 1. Ensure DynamoDB Table
    try:
        dynamodb.describe_table(TableName=DYNAMODB_TABLE_NAME)
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"[AWS] Creating DynamoDB table '{DYNAMODB_TABLE_NAME}'...")
            dynamodb.create_table(
                TableName=DYNAMODB_TABLE_NAME,
                KeySchema=[
                    {'AttributeName': 'alert_id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'alert_id', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            # Wait for table to be active
            waiter = dynamodb.get_waiter('table_exists')
            waiter.wait(TableName=DYNAMODB_TABLE_NAME)

    # 2. Ensure S3 Bucket
    try:
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            print(f"[AWS] Creating S3 Bucket '{S3_BUCKET_NAME}'...")
            # us-east-1 does not require LocationConstraint
            if AWS_REGION == "us-east-1":
                s3.create_bucket(Bucket=S3_BUCKET_NAME)
            else:
                s3.create_bucket(
                    Bucket=S3_BUCKET_NAME,
                    CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                )


def push_alert_to_aws(
    device_id: str, 
    label: str, 
    confidence: float, 
    timestamp: str, 
    device_name: str = "Unknown Device",
    latitude: float = 0.0,
    longitude: float = 0.0,
    gps_accuracy: float = 0.0,
    image_bytes: bytes = None
) -> dict:
    """
    Push a single disaster alert to AWS.
    If image_bytes is provided, uploads it to S3 and attaches the URL to DynamoDB.
    
    Returns a dict with success status and error message (if any).
    """
    dynamodb, s3 = _get_aws_clients()
    if not dynamodb:
        return {"success": False, "error": "AWS Credentials not found or invalid."}

    try:
        # Create infrastructure on first run
        ensure_aws_infrastructure(dynamodb, s3)
        
        alert_id = str(uuid.uuid4())
        image_url = "None"
        
        # Upload image to S3 if provided
        if image_bytes:
            filename = f"alerts/{device_id}_{timestamp.replace(':', '').replace(' ', '_')}.jpg"
            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=filename,
                Body=image_bytes,
                ContentType="image/jpeg"
            )
            # Generate a public-facing URL (assumes bucket is public or uses presigned URL later)
            image_url = f"s3://{S3_BUCKET_NAME}/{filename}"

        # Push metadata to DynamoDB
        dynamodb.put_item(
            TableName=DYNAMODB_TABLE_NAME,
            Item={
                'alert_id': {'S': alert_id},
                'device_id': {'S': device_id},
                'device_name': {'S': device_name},
                'label': {'S': label},
                'confidence': {'N': str(round(confidence, 4))},
                'latitude': {'N': str(latitude) if latitude is not None else '0'},
                'longitude': {'N': str(longitude) if longitude is not None else '0'},
                'gps_accuracy': {'N': str(gps_accuracy) if gps_accuracy is not None else '0'},
                'timestamp': {'S': timestamp},
                'synced_at': {'S': time.strftime("%Y-%m-%d %H:%M:%S")},
                'image_s3_url': {'S': image_url}
            }
        )
        return {"success": True, "alert_id": alert_id, "image_url": image_url}

    except Exception as e:
        return {"success": False, "error": str(e)}

def fetch_aws_alerts(limit=50):
    """Fetch the latest alerts from DynamoDB for the dashboard."""
    dynamodb, s3 = _get_aws_clients()
    if not dynamodb:
        return []

    try:
        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_NAME,
            Limit=limit
        )
        items = response.get('Items', [])
        
        # Parse DynamoDB JSON back to normal dicts
        parsed = []
        for i in items:
            parsed.append({
                "alert_id": i.get('alert_id', {}).get('S', ''),
                "device_id": i.get('device_id', {}).get('S', 'unknown'),
                "device_name": i.get('device_name', {}).get('S', 'Unknown Device'),
                "label": i.get('label', {}).get('S', ''),
                "confidence": float(i.get('confidence', {}).get('N', 0)),
                "latitude": float(i.get('latitude', {}).get('N', 0)),
                "longitude": float(i.get('longitude', {}).get('N', 0)),
                "gps_accuracy": float(i.get('gps_accuracy', {}).get('N', 0)),
                "timestamp": i.get('timestamp', {}).get('S', ''),
                "synced_at": i.get('synced_at', {}).get('S', ''),
                "image_s3_url": i.get('image_s3_url', {}).get('S', '')
            })
        
        # Sort by timestamp descending
        parsed.sort(key=lambda x: x["timestamp"], reverse=True)
        return parsed
    except Exception as e:
        print(f"[AWS] Error fetching alerts: {e}")
        return []

def get_presigned_url(s3_url: str, expiration=3600) -> str:
    """Generate a presigned URL for an S3 object to display it in the UI."""
    if not s3_url or not s3_url.startswith("s3://"):
        return None
        
    dynamodb, s3 = _get_aws_clients()
    if not s3:
        return None
        
    try:
        # Parse s3://bucket/key
        parts = s3_url.replace("s3://", "").split("/", 1)
        if len(parts) != 2:
            return None
        bucket, key = parts
        
        response = s3.generate_presigned_url('get_object',
                                            Params={'Bucket': bucket,
                                                    'Key': key},
                                            ExpiresIn=expiration)
        return response
    except Exception as e:
        print(f"[AWS] Error generating presigned URL: {e}")
        return None
