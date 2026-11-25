"""Client S3 dell'emulatore AWS, mai indirizzato ad endpoint AWS reali."""

from __future__ import annotations

import re

from .ProviderEnvironment import ProviderEnvironmentError


def s3_client(endpoint: str):
    """Costruisce un client S3 esplicitamente collegato al loopback locale.

    La stessa API client puo' essere riutilizzata dai futuri adattatori provider.
    Le prove di funzionamento si trovano esclusivamente sotto tests/.
    """
    if not re.fullmatch(r"http://127\.0\.0\.1:\d{2,5}", endpoint):
        raise ValueError("L'endpoint di controllo S3 deve essere locale su loopback")
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ProviderEnvironmentError(
            "boto3 non installato nella .venv locale: installarlo prima della prova S3"
        ) from exc
    return boto3.client(
        "s3", endpoint_url=endpoint, region_name="us-east-1",
        aws_access_key_id="test", aws_secret_access_key="test",
        config=Config(
            s3={"addressing_style": "path"},
            connect_timeout=3, read_timeout=10,
            retries={"max_attempts": 1},
        ),
    )
