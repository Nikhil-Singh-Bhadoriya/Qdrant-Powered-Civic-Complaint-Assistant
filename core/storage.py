from __future__ import annotations
import hashlib, os, mimetypes
from dataclasses import dataclass
from typing import Optional
from .config import OBJECT_STORE, LOCAL_STORE_DIR, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET

@dataclass
class StoredObject:
    uri: str
    sha256: str
    content_type: str
    size_bytes: int

class ObjectStore:
    def put_bytes(self, data: bytes, filename: str, content_type: Optional[str] = None) -> StoredObject:
        raise NotImplementedError

    def get_local_path(self, uri: str) -> str:
        """Return a local filesystem path for a stored object.
        For remote stores, download to a temp file and return the temp path.
        """
        raise NotImplementedError

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

class LocalObjectStore(ObjectStore):
    def __init__(self, base_dir: str = LOCAL_STORE_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def put_bytes(self, data: bytes, filename: str, content_type: Optional[str] = None) -> StoredObject:
        digest = _sha256(data)
        ext = os.path.splitext(filename)[1] or ""
        path = os.path.join(self.base_dir, digest[:2], digest + ext)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        ct = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return StoredObject(uri=path, sha256=digest, content_type=ct, size_bytes=len(data))

    def get_local_path(self, uri: str) -> str:
        return uri

class MinioObjectStore(ObjectStore):
    def __init__(self):
        import boto3
        from botocore.client import Config
        self.s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.bucket = MINIO_BUCKET
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception:
            self.s3.create_bucket(Bucket=self.bucket)

    def put_bytes(self, data: bytes, filename: str, content_type: Optional[str] = None) -> StoredObject:
        digest = _sha256(data)
        key = f"{digest[:2]}/{digest}/{filename}"
        ct = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=ct)
        return StoredObject(uri=f"s3://{self.bucket}/{key}", sha256=digest, content_type=ct, size_bytes=len(data))

    def get_local_path(self, uri: str) -> str:
        # uri format: s3://bucket/key
        import tempfile, os
        if not uri.startswith('s3://'):
            raise ValueError('Invalid S3 uri')
        parts = uri[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        fd, path = tempfile.mkstemp(prefix='civicfix_', suffix=os.path.basename(key)[:50])
        os.close(fd)
        with open(path, 'wb') as f:
            self.s3.download_fileobj(bucket, key, f)
        return path

def get_object_store() -> ObjectStore:
    if OBJECT_STORE == "minio":
        return MinioObjectStore()
    return LocalObjectStore()
