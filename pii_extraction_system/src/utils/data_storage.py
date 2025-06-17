"""Data storage utilities for local and S3 storage."""

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from core.config import settings
from core.logging_config import get_logger, audit_log

logger = get_logger(__name__)


class DataStorageInterface(ABC):
    """Abstract interface for data storage operations."""
    
    @abstractmethod
    def upload_file(self, local_path: Union[str, Path], remote_path: str) -> bool:
        """Upload a file to storage."""
        pass
    
    @abstractmethod
    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        """Download a file from storage."""
        pass
    
    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        """Delete a file from storage."""
        pass
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in storage."""
        pass
    
    @abstractmethod
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in storage."""
        pass
    
    @abstractmethod
    def get_file_info(self, remote_path: str) -> Optional[Dict]:
        """Get file metadata."""
        pass


class LocalDataStorage(DataStorageInterface):
    """Local file system storage implementation."""
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize local storage with base path."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage initialized at: {self.base_path}")
    
    def _get_full_path(self, remote_path: str) -> Path:
        """Get full local path for a remote path."""
        return self.base_path / remote_path.lstrip('/')
    
    def upload_file(self, local_path: Union[str, Path], remote_path: str) -> bool:
        """Copy file to local storage."""
        try:
            local_path = Path(local_path)
            target_path = self._get_full_path(remote_path)
            
            # Create target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(str(local_path), str(target_path))
            
            audit_log(f"File uploaded to local storage: {local_path} -> {target_path}")
            logger.info(f"File uploaded: {local_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file {local_path} -> {remote_path}: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        """Copy file from local storage."""
        try:
            source_path = self._get_full_path(remote_path)
            local_path = Path(local_path)
            
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source_path}")
                return False
            
            # Create target directory if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(str(source_path), str(local_path))
            
            audit_log(f"File downloaded from local storage: {source_path} -> {local_path}")
            logger.info(f"File downloaded: {source_path} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file {remote_path} -> {local_path}: {e}")
            return False
    
    def delete_file(self, remote_path: str) -> bool:
        """Delete file from local storage."""
        try:
            file_path = self._get_full_path(remote_path)
            
            if file_path.exists():
                file_path.unlink()
                audit_log(f"File deleted from local storage: {file_path}")
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file {remote_path}: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in local storage."""
        try:
            search_path = self._get_full_path(prefix) if prefix else self.base_path
            
            if search_path.is_file():
                return [str(search_path.relative_to(self.base_path))]
            elif search_path.is_dir():
                files = []
                for file_path in search_path.rglob('*'):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(self.base_path)
                        files.append(str(rel_path))
                return sorted(files)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            return []
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in local storage."""
        return self._get_full_path(remote_path).exists()
    
    def get_file_info(self, remote_path: str) -> Optional[Dict]:
        """Get file metadata."""
        try:
            file_path = self._get_full_path(remote_path)
            
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            return {
                'path': str(file_path.relative_to(self.base_path)),
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'created': stat.st_ctime,
                'is_file': file_path.is_file(),
                'is_dir': file_path.is_dir()
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info for {remote_path}: {e}")
            return None


class S3DataStorage(DataStorageInterface):
    """AWS S3 storage implementation."""
    
    def __init__(self, bucket_name: str, region: str = "us-west-2"):
        """Initialize S3 storage."""
        self.bucket_name = bucket_name
        self.region = region
        
        try:
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            
            # Verify bucket access
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 storage initialized for bucket: {bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"S3 bucket not found: {bucket_name}")
            else:
                logger.error(f"S3 initialization error: {e}")
            raise
    
    def upload_file(self, local_path: Union[str, Path], remote_path: str) -> bool:
        """Upload file to S3."""
        try:
            local_path = Path(local_path)
            
            if not local_path.exists():
                logger.error(f"Local file does not exist: {local_path}")
                return False
            
            # Upload file
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                remote_path
            )
            
            audit_log(f"File uploaded to S3: {local_path} -> s3://{self.bucket_name}/{remote_path}")
            logger.info(f"File uploaded to S3: {local_path} -> s3://{self.bucket_name}/{remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file to S3 {local_path} -> {remote_path}: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        """Download file from S3."""
        try:
            local_path = Path(local_path)
            
            # Create target directory if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.s3_client.download_file(
                self.bucket_name,
                remote_path,
                str(local_path)
            )
            
            audit_log(f"File downloaded from S3: s3://{self.bucket_name}/{remote_path} -> {local_path}")
            logger.info(f"File downloaded from S3: s3://{self.bucket_name}/{remote_path} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file from S3 {remote_path} -> {local_path}: {e}")
            return False
    
    def delete_file(self, remote_path: str) -> bool:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=remote_path
            )
            
            audit_log(f"File deleted from S3: s3://{self.bucket_name}/{remote_path}")
            logger.info(f"File deleted from S3: s3://{self.bucket_name}/{remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file from S3 {remote_path}: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(obj['Key'])
            
            return sorted(files)
            
        except Exception as e:
            logger.error(f"Failed to list S3 files with prefix {prefix}: {e}")
            return []
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=remote_path
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking S3 file existence {remote_path}: {e}")
                return False
    
    def get_file_info(self, remote_path: str) -> Optional[Dict]:
        """Get file metadata from S3."""
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=remote_path
            )
            
            return {
                'path': remote_path,
                'size': response['ContentLength'],
                'modified': response['LastModified'].timestamp(),
                'etag': response['ETag'],
                'content_type': response.get('ContentType', ''),
                'metadata': response.get('Metadata', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get S3 file info for {remote_path}: {e}")
            return None


class DataStorageManager:
    """Manager for data storage operations with configurable backend."""
    
    def __init__(self):
        """Initialize storage manager based on configuration."""
        self.storage: DataStorageInterface
        
        if settings.data_source.source_type == "s3":
            if not settings.data_source.s3_bucket:
                raise ValueError("S3 bucket name not configured")
            
            self.storage = S3DataStorage(
                bucket_name=settings.data_source.s3_bucket,
                region=settings.data_source.s3_region
            )
        else:
            # Default to local storage
            self.storage = LocalDataStorage(settings.data_source.local_path)
        
        logger.info(f"Data storage manager initialized with {settings.data_source.source_type} backend")
    
    def upload_document(self, local_path: Union[str, Path], document_id: str) -> bool:
        """Upload a document for processing."""
        local_path = Path(local_path)
        remote_path = f"documents/raw/{document_id}/{local_path.name}"
        return self.storage.upload_file(local_path, remote_path)
    
    def download_document(self, document_id: str, filename: str, local_path: Union[str, Path]) -> bool:
        """Download a document."""
        remote_path = f"documents/raw/{document_id}/{filename}"
        return self.storage.download_file(remote_path, local_path)
    
    def save_processing_result(self, document_id: str, result: Dict) -> bool:
        """Save processing results."""
        try:
            # Save as JSON
            temp_file = Path(f"/tmp/result_{document_id}.json")
            with open(temp_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            remote_path = f"documents/processed/{document_id}/result.json"
            success = self.storage.upload_file(temp_file, remote_path)
            
            # Clean up temp file
            temp_file.unlink()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save processing result for {document_id}: {e}")
            return False
    
    def load_processing_result(self, document_id: str) -> Optional[Dict]:
        """Load processing results."""
        try:
            temp_file = Path(f"/tmp/result_{document_id}.json")
            remote_path = f"documents/processed/{document_id}/result.json"
            
            if self.storage.download_file(remote_path, temp_file):
                with open(temp_file, 'r') as f:
                    result = json.load(f)
                
                # Clean up temp file
                temp_file.unlink()
                
                return result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to load processing result for {document_id}: {e}")
            return None
    
    def list_documents(self, status: str = "raw") -> List[str]:
        """List documents by status (raw, processed, etc.)."""
        prefix = f"documents/{status}/"
        files = self.storage.list_files(prefix)
        
        # Extract document IDs
        document_ids = set()
        for file_path in files:
            parts = file_path.split('/')
            if len(parts) >= 3:
                document_ids.add(parts[2])  # documents/status/doc_id/...
        
        return sorted(list(document_ids))
    
    def cleanup_old_data(self, retention_days: int = None) -> int:
        """Clean up old data based on retention policy."""
        if retention_days is None:
            retention_days = settings.privacy.data_retention_days
        
        # This would need to be implemented based on specific requirements
        # For now, return 0 as placeholder
        logger.info(f"Data cleanup requested for files older than {retention_days} days")
        return 0


# Global storage manager instance
storage_manager = DataStorageManager()