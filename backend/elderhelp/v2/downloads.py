"""Bounded local administrator acquisition. No PDF hosting or public upload endpoint."""

import hashlib
import ipaddress
import os
import socket
import ssl
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpcore
import pymupdf


def safe_destination(url: str, allowed_domains: list[str]) -> tuple[str, str]:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or parsed.port not in (None, 443)
        or parsed.username
        or parsed.password
        or host not in allowed_domains
    ):
        raise ValueError("Source requires HTTPS and an explicitly approved hostname")
    addresses = {r[4][0] for r in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError("Source resolves to a non-public address")
    return host, sorted(addresses)[0]


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_pdf(path: Path, item) -> None:
    if not item.expected_sha256:
        raise ValueError("An administrator-approved SHA-256 is required")
    if path.stat().st_size > item.max_bytes:
        raise ValueError("PDF exceeds byte limit")
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("Not a PDF")
    if checksum(path) != item.expected_sha256:
        raise ValueError("Checksum changed; review and approve a new manifest revision")
    with pymupdf.open(path) as document:
        if document.is_encrypted or not 0 < len(document) <= item.max_pages:
            raise ValueError("Encrypted, empty, or oversized PDF")
        # Touch every page to detect a malformed page tree before promotion.
        for page in document:
            _ = page.rect


def acquire(item, repository_root: Path, seed_root: Path) -> Path:
    root = (repository_root / seed_root).resolve()
    path = (repository_root / item.local_path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Local PDF path is outside the ingestion directory")
    if path.exists():
        validate_pdf(path, item)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            url = str(item.source_url)
            for _ in range(6):
                host, address = safe_destination(url, item.allowed_domains)
                # Connect to the already-validated IP, retaining hostname TLS verification/SNI.
                # This prevents DNS rebinding between validation and connection; proxies are unused.
                parsed = urlsplit(url)
                authority = f"[{address}]" if ":" in address else address
                pinned = f"https://{authority}{parsed.path or '/'}"
                if parsed.query:
                    pinned += "?" + parsed.query
                with httpcore.ConnectionPool(ssl_context=ssl.create_default_context()) as pool:
                    with pool.stream(
                        "GET",
                        pinned,
                        headers={"Host": host, "Accept-Encoding": "identity"},
                        extensions={
                            "sni_hostname": host,
                            "timeout": {"connect": 10, "read": 30, "write": 10, "pool": 10},
                        },
                    ) as response:
                        if response.status in (301, 302, 303, 307, 308):
                            location = dict(response.headers).get(b"location")
                            if not location:
                                raise ValueError("Redirect without location")
                            url = urljoin(url, location.decode("ascii"))
                            continue
                        if response.status != 200:
                            raise ValueError("Publisher download failed")
                        size = 0
                        for block in response.iter_stream():
                            size += len(block)
                            if size > item.max_bytes:
                                raise ValueError("Download exceeds byte limit")
                            output.write(block)
                        break
            else:
                raise ValueError("Too many publisher redirects")
            output.flush()
            os.fsync(output.fileno())
        validate_pdf(temporary, item)
        os.replace(temporary, path)
        temporary = None
        return path
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
