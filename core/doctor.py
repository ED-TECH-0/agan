"""
brain doctor — diagnoses installation and machine.
"""
from dataclasses import dataclass
from typing import Optional
import platform
import subprocess
import sys
import shutil

@dataclass
class DoctorReport:
    profile_match: bool
    recommended_profile: str
    current_profile: str
    issues: list[str]
    warnings: list[str]
    info: dict

class BrainDoctor:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = {}
    
    def check_hardware(self) -> dict:
        import psutil
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        gpu_info = "Not detected"
        vram_gb = 0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = f"{gpu.name} ({gpu.memoryTotal} MB VRAM)"
                vram_gb = gpu.memoryTotal / 1024
        except Exception:
            pass
        
        info = {
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_gb": round(mem.total / (1024**3), 1),
            "ram_available_gb": round(mem.available / (1024**3), 1),
            "gpu": gpu_info,
            "vram_gb": round(vram_gb, 1),
            "disk_free_gb": round(disk.free / (1024**3), 1),
            "os": platform.system(),
            "os_version": platform.version(),
            "arch": platform.machine(),
        }
        self.info.update(info)
        return info
    
    def check_swap_pressure(self):
        import psutil
        swap = psutil.swap_memory()
        if swap.percent > 50:
            self.warnings.append(f"High swap usage: {swap.percent:.1f}% — consider more RAM or smaller model")
    
    def check_thermal(self):
        pass
    
    def check_profile_match(self, current_profile: str) -> tuple[str, bool]:
        import psutil
        mem = psutil.virtual_memory()
        ram_gb = mem.total / (1024**3)
        
        vram_gb = 0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                vram_gb = gpus[0].memoryTotal / 1024
        except Exception:
            pass
        
        if vram_gb >= 24:
            recommended = "max"
        elif vram_gb >= 12:
            recommended = "pro"
        elif vram_gb >= 6 or ram_gb >= 16:
            recommended = "standard"
        elif ram_gb >= 8:
            recommended = "lite"
        else:
            recommended = "nano"
        
        match = current_profile == recommended
        if not match:
            self.warnings.append(f"Current profile '{current_profile}' != recommended '{recommended}'")
        return recommended, match
    
    def check_model_checksums(self):
        pass
    
    def check_searxng(self):
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 8080))
            sock.close()
            if result != 0:
                self.warnings.append("SearXNG not reachable on localhost:8080")
            else:
                self.info["searxng"] = "OK"
        except Exception:
            self.warnings.append("Could not check SearXNG")
    
    def check_sandbox(self):
        try:
            result = subprocess.run([sys.executable, '-c', 'print("ok")'], capture_output=True, timeout=5)
            if result.returncode != 0:
                self.issues.append("Python sandbox not working")
            else:
                self.info["python_sandbox"] = "OK"
        except Exception:
            self.issues.append("Python sandbox failed")
    
    def check_tool_bridges(self):
        tools = {
            "blender": "blender --version",
            "unreal": "UnrealEditor -version",
            "verilator": "verilator --version",
            "iverilog": "iverilog -V",
            "kicad": "kicad --version",
        }
        for name, cmd in tools.items():
            try:
                result = subprocess.run(cmd.split(), capture_output=True, timeout=5)
                if result.returncode == 0:
                    self.info[f"tool_{name}"] = "OK"
                else:
                    self.info[f"tool_{name}"] = "Not found"
            except Exception:
                self.info[f"tool_{name}"] = "Not found"
    
    def check_permissions(self):
        import os
        for path in ["cache", "models", "adapters"]:
            try:
                os.makedirs(path, exist_ok=True)
                test_file = os.path.join(path, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.info[f"perm_{path}"] = "OK"
            except Exception:
                self.issues.append(f"No write permission for {path}/")
    
    def run(self, current_profile: str = "standard", redact: bool = True) -> DoctorReport:
        self.issues = []
        self.warnings = []
        self.info = {}
        
        hw = self.check_hardware()
        self.check_swap_pressure()
        self.check_thermal()
        recommended, match = self.check_profile_match(current_profile)
        self.check_model_checksums()
        self.check_searxng()
        self.check_sandbox()
        self.check_tool_bridges()
        self.check_permissions()
        
        report = DoctorReport(
            profile_match=match,
            recommended_profile=recommended,
            current_profile=current_profile,
            issues=self.issues,
            warnings=self.warnings,
            info=self.info,
        )
        return report
    
    def format_report(self, report: DoctorReport, redact: bool = True) -> str:
        lines = [
            "=== agan brain doctor ===",
            f"Current profile: {report.current_profile}",
            f"Recommended profile: {report.recommended_profile}",
            f"Profile match: {'YES' if report.profile_match else 'NO'}",
            "",
            "--- Hardware ---",
            f"CPU: {report.info.get('cpu_threads', '?')} threads ({report.info.get('cpu_cores', '?')} cores)",
            f"RAM: {report.info.get('ram_gb', '?')} GB total, {report.info.get('ram_available_gb', '?')} GB available",
            f"GPU: {report.info.get('gpu', '?')}",
            f"VRAM: {report.info.get('vram_gb', '?')} GB",
            f"Disk free: {report.info.get('disk_free_gb', '?')} GB",
            f"OS: {report.info.get('os', '?')} {report.info.get('os_version', '')} ({report.info.get('arch', '?')})",
            "",
        ]
        
        if report.issues:
            lines.append("--- Issues (must fix) ---")
            for issue in report.issues:
                lines.append(f"  X {issue}")
            lines.append("")
        
        if report.warnings:
            lines.append("--- Warnings ---")
            for warn in report.warnings:
                lines.append(f"  ! {warn}")
            lines.append("")
        
        lines.append("--- Tool Bridges ---")
        for k, v in report.info.items():
            if k.startswith("tool_"):
                lines.append(f"  {k[5:]}: {v}")
        
        lines.append("")
        lines.append("--- Permissions ---")
        for k, v in report.info.items():
            if k.startswith("perm_"):
                lines.append(f"  {k[5:]}: {v}")
        
        return "\n".join(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--no-redact", action="store_true")
    args = parser.parse_args()
    
    doctor = BrainDoctor()
    report = doctor.run(args.profile, redact=not args.no_redact)
    output = doctor.format_report(report, redact=not args.no_redact)
    print(output)
    
    if args.report:
        with open("brain_doctor_report.txt", "w") as f:
            f.write(output)
        print("\nReport saved to brain_doctor_report.txt")

if __name__ == "__main__":
    main()