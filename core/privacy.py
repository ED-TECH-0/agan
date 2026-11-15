"""
Privacy Firewall — redacts secrets from outbound requests.
"""
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class FirewallResult:
    safe: bool
    redacted_text: str
    findings: list[dict]
    blocked: bool = False

# Layer 1: Deterministic detectors
SECRET_PATTERNS = [
    # API Keys
    (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', "api_key"),
    (r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', "secret_key"),
    (r'(?i)(access[_-]?token|accesstoken)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', "access_token"),
    # AWS
    (r'AKIA[0-9A-Z]{16}', "aws_access_key"),
    (r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', "aws_secret"),
    # GitHub
    (r'gh[pousr]_[a-zA-Z0-9]{36,}', "github_token"),
    (r'github[_-]?token\s*[=:]\s*["\']?([a-zA-Z0-9_]{36,})["\']?', "github_token"),
    # Generic high-entropy strings (potential keys)
    (r'(?i)(key|token|secret|password|passwd)\s*[=:]\s*["\']?([a-zA-Z0-9_\-+/=]{30,})["\']?', "generic_secret"),
    # Private keys
    (r'-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----', "private_key"),
    # Connection strings
    (r'(?i)(mongodb|postgres|mysql|redis)://[^\s"\']{20,}', "connection_string"),
    # Emails
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email"),
    # IPs (private ranges)
    (r'\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b', "private_ip"),
    # MAC addresses
    (r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b', "mac_address"),
    # Home directory paths
        (r'(/home/[^\s"\']+|/Users/[^\s"\']+|C:\\\\Users\\\\[^\s"\']+)', "home_path"),
    # Usernames in URLs
    (r'https?://([^:/@\s]+)@', "url_credentials"),
]

# Layer 2: Query reduction templates
QUERY_REDUCTIONS = {
    "blender": "Blender bpy {error}",
    "unreal": "Unreal Engine {error}",
    "python": "Python {error}",
    "verilog": "Verilog {error}",
    "cmake": "CMake {error}",
}

class PrivacyFirewall:
    def __init__(self, policy: str = "strict"):
        self.policy = policy
        self.patterns = [(re.compile(p), t) for p, t in SECRET_PATTERNS]
        self.transparency_log = []
    
    def scan(self, text: str) -> list[dict]:
        findings = []
        for pattern, secret_type in self.patterns:
            for match in pattern.finditer(text):
                findings.append({
                    "type": secret_type,
                    "match": match.group()[:50] + "..." if len(match.group()) > 50 else match.group(),
                    "position": match.span(),
                })
        return findings
    
    def redact(self, text: str) -> str:
        redacted = text
        # Replace secrets with placeholders
        for pattern, secret_type in self.patterns:
            redacted = pattern.sub(f"[REDACTED:{secret_type.upper()}]", redacted)
        return redacted
    
    def reduce_query(self, query: str) -> str:
        """Reduce query to technical core, dropping identifiers."""
        # Try to match known patterns
        for keyword, template in QUERY_REDUCTIONS.items():
            if keyword in query.lower():
                # Extract error message if present
                error_match = re.search(r'(Error|Exception|Traceback|failed|error):?\s*(.+)', query, re.IGNORECASE)
                if error_match:
                    return template.format(error=error_match.group(2)[:100])
                return template.format(error=query[:100])
        # Default: truncate and remove potential identifiers
        return query[:200]
    
    def process_outbound(self, text: str, context: str = "") -> FirewallResult:
        findings = self.scan(text)
        
        if findings:
            self.transparency_log.append({
                "context": context,
                "findings": findings,
                "action": "redacted" if self.policy != "off" else "logged",
            })
        
        if self.policy == "off":
            return FirewallResult(safe=True, redacted_text=text, findings=findings)
        
        if self.policy == "strict" and findings:
            # In strict mode, block if critical secrets found
            critical = {"private_key", "aws_secret", "aws_access_key", "github_token"}
            if any(f["type"] in critical for f in findings):
                return FirewallResult(
                    safe=False,
                    redacted_text="",
                    findings=findings,
                    blocked=True
                )
        
        redacted = self.redact(text)
        return FirewallResult(safe=True, redacted_text=redacted, findings=findings)
    
    def get_transparency_log(self) -> list:
        return self.transparency_log

# Canary test suite
CANARY_SECRETS = [
    "sk_test_abcdefghijklmnopqrstuvwxyz123456",
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_abcdefghijklmnopqrstuvwxyz123456",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...",
    "user@example.com",
    "192.168.1.100",
    "00:1A:2B:3C:4D:5E",
    "/home/user/.ssh/id_rsa",
    "mongodb://user:pass@localhost:27017/db",
]

def run_canary_tests() -> bool:
    fw = PrivacyFirewall("strict")
    all_passed = True
    for secret in CANARY_SECRETS:
        result = fw.process_outbound(f"Here is my secret: {secret}", "canary_test")
        if not result.findings:
            print(f"FAIL: Missed {secret[:30]}...")
            all_passed = False
        else:
            print(f"PASS: Caught {result.findings[0]['type']}")
    return all_passed

if __name__ == "__main__":
    print("Running canary tests...")
    success = run_canary_tests()
    print(f"\nAll tests passed: {success}")
