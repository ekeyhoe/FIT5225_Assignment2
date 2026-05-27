"""
This file contains the Lambda Handler foe the Tagging Functionality.
It is responsible for handling the incoming requests, processing the data, 
and returning the appropriate responses.

reason for separation of ml_pipeline.py and lambda handler is to ensure that the 
ml_pipeline.py can be reused by the other lambda handler, and to keep the code organized 
and maintainable. 

The ml_pipeline.py contains the core logic for processing the 
data and generating tags, while the lambda handler is responsible for handling 
the incoming requests and returning the responses. 

ml_pipeline_lambda.py = Pure ML

Detects animals
Classifies species
Maps names
Returns tags + detections

tagging_handler.py = Orchestration + I/O

Downloads from S3
Calls ML pipeline
Uploads results (thumbnail, updates DB)
Handles AWS interactions

"""

import json
import os
import uuid
from pathlib import Path

import boto3

from ml_pipeline_lambda import WildlifePipelineLambda, calculate_checksum

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

TABLE_NAME = os.getenv('TABLE_NAME', 'wildlife_files')
MEDIA_BUCKET = os.getenv('MEDIA_BUCKET', 'ecolens-wildlife-files')
THUMBNAIL_PREFIX = 'thumbnails/'

table = dynamodb.Table(TABLE_NAME)


def generate_thumbnail(image_path: str, file_id: str) -> str:
    """Resize image to 300x300, upload to S3. Returns thumbnail S3 URL."""
    from PIL import Image                                                       # this import is already there in the backend ml_pipeline.py file. 
    
    img = Image.open(image_path).convert('RGB')
    img.thumbnail((300, 300), Image.Resampling.LANCZOS)
    
    thumb_path = Path(image_path).parent / f"{file_id}_thumb.jpg"
    img.save(str(thumb_path), 'JPEG', quality=80)
    
    # Upload to S3
    thumb_key = f"{THUMBNAIL_PREFIX}{file_id}_thumb.jpg"
    s3.upload_file(str(thumb_path), MEDIA_BUCKET, thumb_key)
    
    thumb_url = f"s3://{MEDIA_BUCKET}/{thumb_key}"
    
    # Cleanup
    thumb_path.unlink(missing_ok=True)
    
    return thumb_url

def generate_video_thumbnail(video_path: str, file_id: str, pipeline) -> str:      
    """Extract first frame using pipeline, apply image thumbnail function."""
    frame_path = pipeline.extract_first_frame(video_path)
    if not frame_path:
        return None
    return generate_thumbnail(frame_path, file_id)


def lambda_handler(event, context):
    """This function is triggered by the s3 event when a new file is uploaded."""
    
    pipeline = None
    temp_file = None
    
    try:
        records = event.get('Records', [])                                              # trigger form the s3 upload event is registered. 
        if not records:
            return {'statusCode': 400, 'body': json.dumps({'error': 'No S3 event'})}
        
        bucket = records[0]['s3']['bucket']['name']
        key = records[0]['s3']['object']['key']
        

        file_ext = Path(key).suffix.lower()
        is_image = file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv']
        
        if not (is_image or is_video):                                                                  # Neither Image nor Video files throws error. 
            return {'statusCode': 400, 'body': json.dumps({'error': 'Unsupported file type'})}
        
        
        temp_file = f"/tmp/{uuid.uuid4()}{file_ext}"                                    # Download file from S3
        s3.download_file(bucket, key, temp_file)
        
        
        checksum = calculate_checksum(temp_file)                                        # Calculate checksum for deduplication
        
        
        response = table.query(                                                         # Check for duplicate
            IndexName='checksum-index',
            KeyConditionExpression='checksum = :c',
            ExpressionAttributeValues={':c': checksum}
        )
        
        if response['Count'] > 0:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Duplicate file detected. Skipped.',
                    'file_id': response['Items'][0]['file_id']
                })
            }
        
        # Initialize ML pipeline
        pipeline = WildlifePipelineLambda()
        
        # Run ML processing
        if is_image:
            result = pipeline.process_image(temp_file)
        else:
            result = pipeline.process_video(temp_file)                         # 1 frame per second logic is handled in teh ml_pipeline.py
        
        # Generate & upload thumbnail (images and videos)
        file_id = str(uuid.uuid4())
        thumbnail_url = None
        
        if is_image:
            thumbnail_url = generate_thumbnail(temp_file, file_id)
        elif is_video:
            thumbnail_url = generate_video_thumbnail(temp_file, file_id, pipeline)          # Handles Video thumbnail creation using the image thumbnail logic. (first frame = thumbnail)
        
        
        file_url = f"s3://{bucket}/{key}"                   #URL Creation
        
        # Update DynamoDB 
        table.update_item(
            Key={'file_id': file_id},
            UpdateExpression='SET tags = :tags, detections = :dets, thumbnail_url = :thumb, checksum = :cs, file_type = :ft, updated_at = :ts',
            ExpressionAttributeValues={
                ':tags': result['tags'],
                ':dets': result['detections'],
                ':thumb': thumbnail_url or '',
                ':cs': checksum,
                ':ft': result['file_type'],
                ':ts': int(__import__('time').time())
            }
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'File processed and tagged successfully',
                'file_id': file_id,
                'tags': result['tags'],
                'file_url': file_url,
                'thumbnail_url': thumbnail_url
            })
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
    
    finally:                                        # Cleanup temp file and ML pipeline resources
        
        if temp_file and Path(temp_file).exists():
            Path(temp_file).unlink()
        
        if pipeline:                        
            pipeline.cleanup_temp()





