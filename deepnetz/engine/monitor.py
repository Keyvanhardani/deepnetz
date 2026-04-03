"""
Hardware Monitor — real-time system stats for dashboard and adaptive cache.
"""

import time
import subprocess
import platform
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class SystemStats:
    """Snapshot of system resource usage."""
    timestamp: float = 0.0
    # CPU
    cpu_percent: float = 0.0
    cpu_cores: int = 0
    # RAM
    ram_total_mb: int = 0
    ram_used_mb: int = 0
    ram_percent: float = 0.0
    # GPU
    gpu_name: str = ""
    gpu_util_percent: float = 0.0
    gpu_vram_total_mb: int = 0
    gpu_vram_used_mb: int = 0
    gpu_vram_percent: float = 0.0
    gpu_temp_c: float = 0.0
    gpu_power_w: float = 0.0
    # Process
    process_ram_mb: float = 0.0
    # Inference
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cpu": {"percent": self.cpu_percent, "cores": self.cpu_cores},
            "ram": {"total_mb": self.ram_total_mb, "used_mb": self.ram_used_mb,
                    "percent": self.ram_percent},
            "gpu": {"name": self.gpu_name, "util_percent": self.gpu_util_percent,
                    "vram_total_mb": self.gpu_vram_total_mb,
                    "vram_used_mb": self.gpu_vram_used_mb,
                    "vram_percent": self.gpu_vram_percent,
                    "temp_c": self.gpu_temp_c, "power_w": self.gpu_power_w},
            "process": {"ram_mb": self.process_ram_mb},
            "inference": {"tokens": self.tokens_generated,
                          "tps": self.tokens_per_sec},
        }


class SystemMonitor:
    """Collects system stats. Call get_stats() for a snapshot."""

    def __init__(self):
        self._psutil = None
        self._process = None
        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process()
        except ImportError:
            pass

    def get_stats(self) -> SystemStats:
        stats = SystemStats(timestamp=time.time())

        if self._psutil:
            # CPU
            stats.cpu_percent = self._psutil.cpu_percent(interval=0)
            stats.cpu_cores = self._psutil.cpu_count() or 0

            # RAM
            mem = self._psutil.virtual_memory()
            stats.ram_total_mb = int(mem.total / (1024 * 1024))
            stats.ram_used_mb = int(mem.used / (1024 * 1024))
            stats.ram_percent = mem.percent

            # Process RAM
            try:
                pi = self._process.memory_info()
                stats.process_ram_mb = pi.rss / (1024 * 1024)
            except Exception:
                pass
        else:
            # Fallback without psutil
            stats.cpu_cores = __import__("os").cpu_count() or 0
            if platform.system() == "Linux":
                try:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                stats.ram_total_mb = int(line.split()[1]) // 1024
                            elif line.startswith("MemAvailable:"):
                                avail = int(line.split()[1]) // 1024
                                stats.ram_used_mb = stats.ram_total_mb - avail
                                stats.ram_percent = 100 * stats.ram_used_mb / max(stats.ram_total_mb, 1)
                except Exception:
                    pass

        # GPU via nvidia-smi
        self._read_gpu_stats(stats)

        return stats

    def _read_gpu_stats(self, stats: SystemStats):
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,utilization.gpu,memory.total,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 6:
                    stats.gpu_name = parts[0]
                    stats.gpu_util_percent = float(parts[1]) if parts[1] != "[N/A]" else 0
                    stats.gpu_vram_total_mb = int(float(parts[2]))
                    stats.gpu_vram_used_mb = int(float(parts[3]))
                    stats.gpu_vram_percent = 100 * stats.gpu_vram_used_mb / max(stats.gpu_vram_total_mb, 1)
                    stats.gpu_temp_c = float(parts[4]) if parts[4] != "[N/A]" else 0
                    stats.gpu_power_w = float(parts[5]) if parts[5] != "[N/A]" else 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass


# Global singleton
_monitor = None

def get_monitor() -> SystemMonitor:
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    return _monitor
