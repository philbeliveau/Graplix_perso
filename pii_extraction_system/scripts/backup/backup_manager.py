#!/usr/bin/env python3
"""Comprehensive backup and recovery manager for the PII extraction system."""

import os
import shutil
import sqlite3
import subprocess
import gzip
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from loguru import logger
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


class BackupType(Enum):
    """Backup types."""
    DATABASE = "database"
    FILES = "files" 
    CONFIG = "config"
    MODELS = "models"
    LOGS = "logs"
    FULL = "full"


class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackupInfo:
    """Backup information."""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    status: BackupStatus
    file_path: str
    size_bytes: int
    checksum: str
    retention_date: datetime
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class BackupManager:
    """Comprehensive backup and recovery manager."""
    
    def __init__(self, backup_dir: Path = None, s3_bucket: str = None):
        """Initialize backup manager."""
        self.backup_dir = backup_dir or Path("backups")
        self.s3_bucket = s3_bucket
        self.backup_history: List[BackupInfo] = []
        self.s3_client = None
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client if bucket specified
        if self.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                logger.info(f"S3 backup enabled to bucket: {self.s3_bucket}")
            except NoCredentialsError:
                logger.warning("AWS credentials not found. S3 backup disabled.")
        
        # Load backup history
        self._load_backup_history()
    
    def _load_backup_history(self) -> None:
        """Load backup history from file."""
        history_file = self.backup_dir / "backup_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                self.backup_history = []
                for item in data:
                    backup_info = BackupInfo(
                        backup_id=item["backup_id"],
                        backup_type=BackupType(item["backup_type"]),
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        status=BackupStatus(item["status"]),
                        file_path=item["file_path"],
                        size_bytes=item["size_bytes"],
                        checksum=item["checksum"],
                        retention_date=datetime.fromisoformat(item["retention_date"]),
                        metadata=item["metadata"],
                        error_message=item.get("error_message")
                    )
                    self.backup_history.append(backup_info)
                
                logger.info(f"Loaded {len(self.backup_history)} backup records")
            
            except Exception as e:
                logger.error(f"Failed to load backup history: {e}")
    
    def _save_backup_history(self) -> None:
        """Save backup history to file."""
        history_file = self.backup_dir / "backup_history.json"
        
        try:
            data = []
            for backup_info in self.backup_history:
                item = asdict(backup_info)
                item["backup_type"] = backup_info.backup_type.value
                item["status"] = backup_info.status.value
                item["timestamp"] = backup_info.timestamp.isoformat()
                item["retention_date"] = backup_info.retention_date.isoformat()
                data.append(item)
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save backup history: {e}")
    
    def _generate_backup_id(self, backup_type: BackupType) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{backup_type.value}_{timestamp}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _compress_file(self, source_path: Path, compressed_path: Path) -> None:
        """Compress file using gzip."""
        with open(source_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def _upload_to_s3(self, local_path: Path, s3_key: str) -> bool:
        """Upload file to S3."""
        if not self.s3_client or not self.s3_bucket:
            return False
        
        try:
            self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_key}")
            return True
        
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
    
    def backup_database(self, db_path: str = None, retention_days: int = 30) -> BackupInfo:
        """Backup database."""
        backup_id = self._generate_backup_id(BackupType.DATABASE)
        backup_file = self.backup_dir / f"{backup_id}.db.gz"
        
        # Create backup info
        backup_info = BackupInfo(
            backup_id=backup_id,
            backup_type=BackupType.DATABASE,
            timestamp=datetime.now(),
            status=BackupStatus.RUNNING,
            file_path=str(backup_file),
            size_bytes=0,
            checksum="",
            retention_date=datetime.now() + timedelta(days=retention_days),
            metadata={"db_path": db_path}
        )
        
        try:
            # Default database paths
            if not db_path:
                possible_paths = [
                    "data/pii_extraction.db",
                    "pii_extraction.db",
                    "data/pii_extraction_dev.db"
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        db_path = path
                        break
                
                if not db_path:
                    raise FileNotFoundError("Database file not found")
            
            db_path = Path(db_path)
            if not db_path.exists():
                raise FileNotFoundError(f"Database file not found: {db_path}")
            
            # Create backup
            if db_path.suffix == '.db':
                # SQLite database
                temp_backup = self.backup_dir / f"{backup_id}.db"
                
                # Use SQLite .backup command for consistent backup
                conn = sqlite3.connect(str(db_path))
                backup_conn = sqlite3.connect(str(temp_backup))
                conn.backup(backup_conn)
                conn.close()
                backup_conn.close()
                
                # Compress backup
                self._compress_file(temp_backup, backup_file)
                temp_backup.unlink()
            
            else:
                # Other database types - use direct file copy
                shutil.copy2(db_path, backup_file.with_suffix(''))
                self._compress_file(backup_file.with_suffix(''), backup_file)
                backup_file.with_suffix('').unlink()
            
            # Update backup info
            backup_info.size_bytes = backup_file.stat().st_size
            backup_info.checksum = self._calculate_checksum(backup_file)
            backup_info.status = BackupStatus.COMPLETED
            
            # Upload to S3 if configured
            if self.s3_client:
                s3_key = f"database/{backup_id}.db.gz"
                self._upload_to_s3(backup_file, s3_key)
                backup_info.metadata["s3_key"] = s3_key
            
            logger.info(f"Database backup completed: {backup_id}")
        
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            logger.error(f"Database backup failed: {e}")
        
        # Add to history and save
        self.backup_history.append(backup_info)
        self._save_backup_history()
        
        return backup_info
    
    def backup_files(self, source_dirs: List[str] = None, retention_days: int = 30) -> BackupInfo:
        """Backup files and directories."""
        backup_id = self._generate_backup_id(BackupType.FILES)
        backup_file = self.backup_dir / f"{backup_id}.tar.gz"
        
        # Default directories to backup
        if not source_dirs:
            source_dirs = ["data", "config", "logs"]
        
        # Create backup info
        backup_info = BackupInfo(
            backup_id=backup_id,
            backup_type=BackupType.FILES,
            timestamp=datetime.now(),
            status=BackupStatus.RUNNING,
            file_path=str(backup_file),
            size_bytes=0,
            checksum="",
            retention_date=datetime.now() + timedelta(days=retention_days),
            metadata={"source_dirs": source_dirs}
        )
        
        try:
            # Create tar archive
            import tarfile
            
            with tarfile.open(backup_file, "w:gz") as tar:
                for dir_path in source_dirs:
                    dir_path = Path(dir_path)
                    if dir_path.exists():
                        tar.add(dir_path, arcname=dir_path.name)
                        logger.info(f"Added {dir_path} to backup")
                    else:
                        logger.warning(f"Directory not found: {dir_path}")
            
            # Update backup info
            backup_info.size_bytes = backup_file.stat().st_size
            backup_info.checksum = self._calculate_checksum(backup_file)
            backup_info.status = BackupStatus.COMPLETED
            
            # Upload to S3 if configured
            if self.s3_client:
                s3_key = f"files/{backup_id}.tar.gz"
                self._upload_to_s3(backup_file, s3_key)
                backup_info.metadata["s3_key"] = s3_key
            
            logger.info(f"Files backup completed: {backup_id}")
        
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            logger.error(f"Files backup failed: {e}")
        
        # Add to history and save
        self.backup_history.append(backup_info)
        self._save_backup_history()
        
        return backup_info
    
    def backup_models(self, retention_days: int = 90) -> BackupInfo:
        """Backup ML models."""
        backup_id = self._generate_backup_id(BackupType.MODELS)
        backup_file = self.backup_dir / f"{backup_id}_models.tar.gz"
        
        # Create backup info
        backup_info = BackupInfo(
            backup_id=backup_id,
            backup_type=BackupType.MODELS,
            timestamp=datetime.now(),
            status=BackupStatus.RUNNING,
            file_path=str(backup_file),
            size_bytes=0,
            checksum="",
            retention_date=datetime.now() + timedelta(days=retention_days),
            metadata={}
        )
        
        try:
            model_dirs = []
            possible_model_paths = [
                "data/models",
                "models",
                "src/dashboard/data/models"
            ]
            
            for path in possible_model_paths:
                model_path = Path(path)
                if model_path.exists():
                    model_dirs.append(model_path)
            
            if not model_dirs:
                raise FileNotFoundError("No model directories found")
            
            # Create tar archive
            import tarfile
            
            with tarfile.open(backup_file, "w:gz") as tar:
                for model_dir in model_dirs:
                    tar.add(model_dir, arcname=f"models_{model_dir.name}")
                    logger.info(f"Added {model_dir} to models backup")
            
            # Update backup info
            backup_info.size_bytes = backup_file.stat().st_size
            backup_info.checksum = self._calculate_checksum(backup_file)
            backup_info.status = BackupStatus.COMPLETED
            backup_info.metadata["model_dirs"] = [str(d) for d in model_dirs]
            
            # Upload to S3 if configured
            if self.s3_client:
                s3_key = f"models/{backup_id}_models.tar.gz"
                self._upload_to_s3(backup_file, s3_key)
                backup_info.metadata["s3_key"] = s3_key
            
            logger.info(f"Models backup completed: {backup_id}")
        
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            logger.error(f"Models backup failed: {e}")
        
        # Add to history and save
        self.backup_history.append(backup_info)
        self._save_backup_history()
        
        return backup_info
    
    def full_backup(self, retention_days: int = 30) -> List[BackupInfo]:
        """Perform full system backup."""
        logger.info("Starting full system backup")
        
        backups = []
        
        # Backup database
        try:
            db_backup = self.backup_database(retention_days=retention_days)
            backups.append(db_backup)
        except Exception as e:
            logger.error(f"Database backup failed during full backup: {e}")
        
        # Backup files
        try:
            files_backup = self.backup_files(retention_days=retention_days)
            backups.append(files_backup)
        except Exception as e:
            logger.error(f"Files backup failed during full backup: {e}")
        
        # Backup models
        try:
            models_backup = self.backup_models(retention_days=retention_days)
            backups.append(models_backup)
        except Exception as e:
            logger.error(f"Models backup failed during full backup: {e}")
        
        logger.info(f"Full backup completed with {len(backups)} components")
        return backups
    
    def restore_database(self, backup_id: str, target_path: str = None) -> bool:
        """Restore database from backup."""
        backup_info = self.get_backup_info(backup_id)
        
        if not backup_info or backup_info.backup_type != BackupType.DATABASE:
            logger.error(f"Database backup not found: {backup_id}")
            return False
        
        backup_file = Path(backup_info.file_path)
        
        if not backup_file.exists():
            # Try downloading from S3
            if self.s3_client and "s3_key" in backup_info.metadata:
                s3_key = backup_info.metadata["s3_key"]
                try:
                    self.s3_client.download_file(self.s3_bucket, s3_key, str(backup_file))
                    logger.info(f"Downloaded backup from S3: {s3_key}")
                except ClientError as e:
                    logger.error(f"Failed to download backup from S3: {e}")
                    return False
            else:
                logger.error(f"Backup file not found: {backup_file}")
                return False
        
        try:
            # Decompress backup
            temp_db = backup_file.with_suffix('.db')
            
            with gzip.open(backup_file, 'rb') as f_in:
                with open(temp_db, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Move to target location
            if not target_path:
                target_path = backup_info.metadata.get("db_path", "data/pii_extraction_restored.db")
            
            target_path = Path(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(temp_db, target_path)
            
            logger.info(f"Database restored to: {target_path}")
            return True
        
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    def restore_files(self, backup_id: str, target_dir: str = None) -> bool:
        """Restore files from backup."""
        backup_info = self.get_backup_info(backup_id)
        
        if not backup_info or backup_info.backup_type != BackupType.FILES:
            logger.error(f"Files backup not found: {backup_id}")
            return False
        
        backup_file = Path(backup_info.file_path)
        
        if not backup_file.exists():
            # Try downloading from S3
            if self.s3_client and "s3_key" in backup_info.metadata:
                s3_key = backup_info.metadata["s3_key"]
                try:
                    self.s3_client.download_file(self.s3_bucket, s3_key, str(backup_file))
                    logger.info(f"Downloaded backup from S3: {s3_key}")
                except ClientError as e:
                    logger.error(f"Failed to download backup from S3: {e}")
                    return False
            else:
                logger.error(f"Backup file not found: {backup_file}")
                return False
        
        try:
            # Extract tar archive
            import tarfile
            
            if not target_dir:
                target_dir = "restored_files"
            
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(target_path)
            
            logger.info(f"Files restored to: {target_path}")
            return True
        
        except Exception as e:
            logger.error(f"Files restore failed: {e}")
            return False
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Get backup information by ID."""
        for backup in self.backup_history:
            if backup.backup_id == backup_id:
                return backup
        return None
    
    def list_backups(self, backup_type: BackupType = None) -> List[BackupInfo]:
        """List available backups."""
        if backup_type:
            return [b for b in self.backup_history if b.backup_type == backup_type]
        return self.backup_history.copy()
    
    def cleanup_old_backups(self) -> int:
        """Clean up expired backups."""
        current_time = datetime.now()
        cleaned_count = 0
        
        backups_to_remove = []
        
        for backup in self.backup_history:
            if backup.retention_date < current_time:
                # Remove local file
                backup_file = Path(backup.file_path)
                if backup_file.exists():
                    backup_file.unlink()
                    logger.info(f"Removed expired backup file: {backup_file}")
                
                # Remove from S3 if exists
                if self.s3_client and "s3_key" in backup.metadata:
                    s3_key = backup.metadata["s3_key"]
                    try:
                        self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                        logger.info(f"Removed expired backup from S3: {s3_key}")
                    except ClientError as e:
                        logger.warning(f"Failed to remove S3 backup {s3_key}: {e}")
                
                backups_to_remove.append(backup)
                cleaned_count += 1
        
        # Remove from history
        for backup in backups_to_remove:
            self.backup_history.remove(backup)
        
        # Save updated history
        if backups_to_remove:
            self._save_backup_history()
        
        logger.info(f"Cleaned up {cleaned_count} expired backups")
        return cleaned_count
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics."""
        stats = {
            "total_backups": len(self.backup_history),
            "backup_types": {},
            "total_size": 0,
            "completed_backups": 0,
            "failed_backups": 0,
            "latest_backup": None,
            "oldest_backup": None
        }
        
        if not self.backup_history:
            return stats
        
        # Calculate statistics
        for backup in self.backup_history:
            backup_type = backup.backup_type.value
            stats["backup_types"][backup_type] = stats["backup_types"].get(backup_type, 0) + 1
            stats["total_size"] += backup.size_bytes
            
            if backup.status == BackupStatus.COMPLETED:
                stats["completed_backups"] += 1
            elif backup.status == BackupStatus.FAILED:
                stats["failed_backups"] += 1
        
        # Find latest and oldest
        sorted_backups = sorted(self.backup_history, key=lambda x: x.timestamp)
        stats["oldest_backup"] = sorted_backups[0].timestamp.isoformat()
        stats["latest_backup"] = sorted_backups[-1].timestamp.isoformat()
        
        # Convert size to human readable
        stats["total_size_mb"] = stats["total_size"] / (1024 * 1024)
        
        return stats


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PII Extraction System Backup Manager")
    parser.add_argument("command", choices=["backup", "restore", "list", "cleanup", "stats"])
    parser.add_argument("--type", choices=["database", "files", "models", "full"], default="full")
    parser.add_argument("--backup-id", help="Backup ID for restore operations")
    parser.add_argument("--target", help="Target path for restore operations")
    parser.add_argument("--s3-bucket", help="S3 bucket for cloud backups")
    parser.add_argument("--retention-days", type=int, default=30, help="Backup retention days")
    
    args = parser.parse_args()
    
    # Initialize backup manager
    backup_manager = BackupManager(s3_bucket=args.s3_bucket)
    
    if args.command == "backup":
        if args.type == "database":
            result = backup_manager.backup_database(retention_days=args.retention_days)
            print(f"Database backup: {result.backup_id} - {result.status.value}")
        
        elif args.type == "files":
            result = backup_manager.backup_files(retention_days=args.retention_days)
            print(f"Files backup: {result.backup_id} - {result.status.value}")
        
        elif args.type == "models":
            result = backup_manager.backup_models(retention_days=args.retention_days)
            print(f"Models backup: {result.backup_id} - {result.status.value}")
        
        elif args.type == "full":
            results = backup_manager.full_backup(retention_days=args.retention_days)
            for result in results:
                print(f"{result.backup_type.value} backup: {result.backup_id} - {result.status.value}")
    
    elif args.command == "restore":
        if not args.backup_id:
            print("Error: --backup-id required for restore operations")
            return
        
        if args.type == "database":
            success = backup_manager.restore_database(args.backup_id, args.target)
            print(f"Database restore: {'Success' if success else 'Failed'}")
        
        elif args.type == "files":
            success = backup_manager.restore_files(args.backup_id, args.target)
            print(f"Files restore: {'Success' if success else 'Failed'}")
    
    elif args.command == "list":
        backups = backup_manager.list_backups()
        print(f"{'Backup ID':<30} {'Type':<10} {'Status':<10} {'Size (MB)':<10} {'Date':<20}")
        print("-" * 80)
        
        for backup in sorted(backups, key=lambda x: x.timestamp, reverse=True):
            size_mb = backup.size_bytes / (1024 * 1024)
            print(f"{backup.backup_id:<30} {backup.backup_type.value:<10} {backup.status.value:<10} {size_mb:<10.2f} {backup.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    elif args.command == "cleanup":
        count = backup_manager.cleanup_old_backups()
        print(f"Cleaned up {count} expired backups")
    
    elif args.command == "stats":
        stats = backup_manager.get_backup_statistics()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()