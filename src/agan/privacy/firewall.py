"""Outbound privacy gateway for AGAN.

Blocks loopback/private IP destinations and redacts API keys from text.
"""
from __future__ import annotations

import ipaddress
import re
from typing import Optional

# Common API key prefixes/patterns to redact
_API_KEY_PATTERNS = [
    # OpenAI-style keys: sk-...
    re.compile(r'sk-[A-Za-z0-9]{20,}'),
    # AWS access keys: AKIA...
    re.compile(r'AKIA[0-9A-Z]{16}'),
    # Slack tokens: xoxb-, xoxp-, xoxa-
    re.compile(r'xox[bap]-[A-Za-z0-9-]+'),
    # Generic "key =" / "token =" assignments
    re.compile(r'(api[_-]?key|secret|token|password)\s*[:=]\s*["\']?([A-Za-z0-9_-]{8,})["\']?', re.IGNORECASE),
    # Bearer tokens
    re.compile(r'Bearer\s+[A-Za-z0-9._-]{10,}'),
]

# Loopback and private IP ranges to block
_BLOCKED_NETWORKS = [
    ipaddress.ip_network('127.0.0.0/8'),      # IPv4 loopback
    ipaddress.ip_network('::1/128'),           # IPv6 loopback
    ipaddress.ip_network('fe80::/10'),         # IPv6 link-local
    ipaddress.ip_network('10.0.0.0/8'),        # RFC1918 private
    ipaddress.ip_network('172.16.0.0/12'),     # RFC1918 private
    ipaddress.ip_network('192.168.0.0/16'),    # RFC1918 private
    ipaddress.ip_network('0.0.0.0/8'),         # "this network"
    ipaddress.ip_network('169.254.0.0/16'),    # link-local
    ipaddress.ip_network('224.0.0.0/4'),       # multicast
    ipaddress.ip_network('240.0.0.0/4'),       # reserved
]


def _is_blocked_host(host: str) -> bool:
    """Return True if ``host`` resolves to a blocked/private IP."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Not a literal IP; try DNS resolution
        try:
            import socket
            infos = socket.getaddrinfo(host, None)
            for info in infos:
                sockaddr = info[4]
                if sockaddr:
                    try:
                        ip = ipaddress.ip_address(sockaddr[0])
                    except ValueError:
                        continue
                    if any(ip in net for net in _BLOCKED_NETWORKS):
                        return True
            return False
        except OSError:
            return False
    return any(addr in net for net in _BLOCKED_NETWORKS)


def redact_api_keys(text: str) -> str:
    """Redact API keys and secrets from ``text``, returning sanitized text."""
    result = text
    for pattern in _API_KEY_PATTERNS:
        result = pattern.sub(lambda m: '[REDACTED]', result)
    return result


class OutboundGateway:
    """Gatekeeper for outbound data flows respecting privacy boundaries."""

    def __init__(self, block_loopback: bool = True, block_private: bool = True):
        self.block_loopback = block_loopback
        self.block_private = block_private

    def check(self, question: str) -> bool:
        """Return True if the question passes privacy checks.

        Blocks requests that target loopback or private IP addresses.
        """
        # Extract any IP-looking or host-looking tokens from the question
        tokens = re.findall(r'\b(?:https?://)?([A-Za-z0-9.-]+)\b', question)
        for token in tokens:
            try:
                ipaddress.ip_address(token)
                is_ip = True
            except ValueError:
                is_ip = False
            if is_ip or '.' in token:
                if self.block_loopback or self.block_private:
                    if _is_blocked_host(token):
                        return False
        return True

    def redact_sensitive(self, text: str) -> str:
        """Remove or mask sensitive references from ``text``."""
        return redact_api_keys(text)

    def check_url(self, url: str) -> bool:
        """Return True if the URL's host is allowed (not blocked)."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname or ''
        except Exception:
            return False
        if self.block_loopback or self.block_private:
            if _is_blocked_host(host):
                return False
        return True


__all__ = [
    "OutboundGateway",
    "redact_api_keys",
    "_is_blocked_host",
    "_BLOCKED_NETWORKS",
    "_API_KEY_PATTERNS",
]
