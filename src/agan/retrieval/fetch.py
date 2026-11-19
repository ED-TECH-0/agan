"""HTTP fetching with SSRF protection, robots.txt compliance, and rate limiting."""
import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urljoin
import ipaddress

import httpx
import trafilatura
from trafilatura.settings import use_config

from agan.schemas import Chunk

# Global rate limiter per domain
_domain_locks: dict[str, asyncio.Lock] = {}
_domain_last_request: dict[str, float] = {}
_DOMAIN_RPS = 1.0  # 1 request per second per domain
_REQUEST_TIMEOUT = 15.0
_CONNECT_TIMEOUT = 5.0
_MAX_REDIRECTS = 3
_MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10 MB
_USER_AGENT = "AGAN/0.1 (+https://github.com/ED-TECH-0/agan)"

# Robots.txt cache: domain -> (rules, expiry)
_robots_cache: dict[str, tuple[dict, float]] = {}
_ROBOTS_TTL = 24 * 3600  # 24 hours

# Private IP ranges to block (SSRF protection)
_PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return any(addr in net for net in _PRIVATE_RANGES)
    except ValueError:
        return False


async def _resolve_host(host: str) -> list[str]:
    """Resolve hostname to IPs, checking for private addresses."""
    import socket
    try:
        loop = asyncio.get_event_loop()
        infos = await loop.getaddrinfo(host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
        ips = [info[4][0] for info in infos]
        return [str(ip) for ip in ips]
    except Exception:
        return []


def _check_ssrf(url: str) -> Optional[str]:
    """Check URL for SSRF risks. Returns error message if blocked, None if OK."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Blocked scheme: {parsed.scheme}"

    host = parsed.hostname or ""
    if not host:
        return "No hostname"

    # Check for localhost/loopback in hostname
    if host in ("localhost", "localhost.localdomain", "127.0.0.1", "::1", "[::1]"):
        return "Blocked: localhost"

    # Resolve and check IPs
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    ips = loop.run_until_complete(_resolve_host(host))
    for ip in ips:
        if _is_private_ip(ip):
            return f"Blocked: resolves to private IP {ip}"

    return None


async def _get_robots_rules(domain: str) -> dict:
    """Fetch and parse robots.txt for a domain. Returns dict with 'allow' and 'disallow' patterns."""
    now = time.time()
    if domain in _robots_cache:
        rules, expiry = _robots_cache[domain]
        if now < expiry:
            return rules

    rules = {"allow": [], "disallow": []}
    robots_url = f"https://{domain}/robots.txt"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(robots_url, headers={"User-Agent": _USER_AGENT})
            if resp.status_code == 200:
                current_agent = None
                for line in resp.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("user-agent:"):
                        current_agent = line.split(":", 1)[1].strip()
                    elif current_agent in ("*", _USER_AGENT) or current_agent is None:
                        if line.lower().startswith("disallow:"):
                            path = line.split(":", 1)[1].strip()
                            if path:
                                rules["disallow"].append(path)
                        elif line.lower().startswith("allow:"):
                            path = line.split(":", 1)[1].strip()
                            if path:
                                rules["allow"].append(path)
    except Exception:
        pass  # On error, allow all

    _robots_cache[domain] = (rules, now + _ROBOTS_TTL)
    return rules


def _check_robots(url: str, rules: dict) -> bool:
    """Check if URL is allowed by robots.txt rules."""
    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    # Check disallow patterns (simple prefix matching)
    for pattern in rules.get("disallow", []):
        if path.startswith(pattern):
            # Check if there's a more specific allow rule
            allowed = False
            for allow_pattern in rules.get("allow", []):
                if path.startswith(allow_pattern) and len(allow_pattern) > len(pattern):
                    allowed = True
                    break
            if not allowed:
                return False
    return True


async def _rate_limit(domain: str):
    """Enforce per-domain rate limit."""
    if domain not in _domain_locks:
        _domain_locks[domain] = asyncio.Lock()
    async with _domain_locks[domain]:
        now = time.time()
        last = _domain_last_request.get(domain, 0)
        min_interval = 1.0 / _DOMAIN_RPS
        if now - last < min_interval:
            await asyncio.sleep(min_interval - (now - last))
        _domain_last_request[domain] = time.time()


@dataclass
class FetchResult:
    url: str
    html: str
    final_url: str
    status_code: int
    content_type: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    error: Optional[str] = None


async def fetch_url(url: str, *, conditional: bool = False, etag: Optional[str] = None, last_modified: Optional[str] = None) -> FetchResult:
    """Fetch a URL with SSRF protection, robots.txt, rate limiting, and conditional requests."""
    # SSRF check
    ssrf_error = _check_ssrf(url)
    if ssrf_error:
        return FetchResult(url=url, html="", final_url=url, status_code=0, content_type="", error=ssrf_error)

    parsed = urlparse(url)
    domain = parsed.netloc

    # Robots.txt check
    rules = await _get_robots_rules(domain)
    if not _check_robots(url, rules):
        return FetchResult(url=url, html="", final_url=url, status_code=0, content_type="", error="Blocked by robots.txt")

    # Rate limit
    await _rate_limit(domain)

    # Conditional request headers
    headers = {"User-Agent": _USER_AGENT}
    if conditional and etag:
        headers["If-None-Match"] = etag
    if conditional and last_modified:
        headers["If-Modified-Since"] = last_modified

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_CONNECT_TIMEOUT, read=_REQUEST_TIMEOUT),
            follow_redirects=True,
            max_redirects=_MAX_REDIRECTS,
            headers=headers,
        ) as client:
            resp = await client.get(url)
            final_url = str(resp.url)

            # Check final URL for SSRF after redirects
            ssrf_error = _check_ssrf(final_url)
            if ssrf_error:
                return FetchResult(url=url, html="", final_url=final_url, status_code=0, content_type="", error=ssrf_error)

            # Check content size
            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) > _MAX_CONTENT_SIZE:
                return FetchResult(url=url, html="", final_url=final_url, status_code=resp.status_code,
                                   content_type=resp.headers.get("content-type", ""), error="Content too large")

            # Check content type - only process HTML/text
            content_type = resp.headers.get("content-type", "").lower()
            if "html" not in content_type and "text" not in content_type:
                return FetchResult(url=url, html="", final_url=final_url, status_code=resp.status_code,
                                   content_type=content_type, error="Non-HTML content")

            html = resp.text
            if len(html) > _MAX_CONTENT_SIZE:
                html = html[:_MAX_CONTENT_SIZE]

            return FetchResult(
                url=url,
                html=html,
                final_url=final_url,
                status_code=resp.status_code,
                content_type=content_type,
                etag=resp.headers.get("etag"),
                last_modified=resp.headers.get("last-modified"),
            )

    except httpx.TimeoutException:
        return FetchResult(url=url, html="", final_url=url, status_code=0, content_type="", error="Timeout")
    except httpx.TooManyRedirects:
        return FetchResult(url=url, html="", final_url=url, status_code=0, content_type="", error="Too many redirects")
    except Exception as e:
        return FetchResult(url=url, html="", final_url=url, status_code=0, content_type="", error=str(e))


async def extract_text(html: str, url: str) -> str:
    """Extract main text content from HTML using trafilatura."""
    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_tables=True,
            favor_precision=True,
            config=config,
        )
        return extracted or ""
    except Exception:
        return ""


async def fetch_and_extract(url: str) -> tuple[str, FetchResult]:
    """Fetch URL and extract text in one call."""
    result = await fetch_url(url)
    if result.error:
        return "", result
    text = await extract_text(result.html, result.final_url)
    return text, result


__all__ = ["fetch_url", "extract_text", "fetch_and_extract", "FetchResult", "_check_ssrf", "_get_robots_rules", "_rate_limit"]