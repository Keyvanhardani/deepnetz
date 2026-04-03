"""
Hardware detection and budget planning.

Detects available GPU(s), VRAM, system RAM, CPU cores,
and recommends optimal configuration for a given model.
"""

import subprocess
import os
import platform
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    compute_capability: str
    index: int


@dataclass
class HardwareProfile:
    gpus: List[GPUInfo]
    total_vram_mb: int
    ram_mb: int
    cpu_cores: int
    os: str
    has_cuda: bool


def detect_gpus() -> List[GPUInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        vram_mb=int(float(parts[2])),
                        compute_capability=parts[3]
                    ))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def detect_ram_mb() -> int:
    """Detect total system RAM in MB."""
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip()) // (1024 * 1024)
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulong),
                    ("ullAvailPhys", c_ulong),
                    ("ullTotalPageFile", c_ulong),
                    ("ullAvailPageFile", c_ulong),
                    ("ullTotalVirtual", c_ulong),
                    ("ullAvailVirtual", c_ulong),
                    ("ullAvailExtendedVirtual", c_ulong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys // (1024 * 1024)
    except Exception:
        pass
    return 16384  # default 16GB


def detect_hardware() -> HardwareProfile:
    """Full hardware detection."""
    gpus = detect_gpus()
    ram = detect_ram_mb()
    cores = os.cpu_count() or 4

    return HardwareProfile(
        gpus=gpus,
        total_vram_mb=sum(g.vram_mb for g in gpus),
        ram_mb=ram,
        cpu_cores=cores,
        os=platform.system(),
        has_cuda=len(gpus) > 0
    )


def print_hardware(hw: HardwareProfile):
    """Pretty-print hardware info."""
    print(f"\n  DeepNetz Hardware Profile")
    print(f"  {'-' * 40}")
    print(f"  OS:       {hw.os}")
    print(f"  CPU:      {hw.cpu_cores} cores")
    print(f"  RAM:      {hw.ram_mb // 1024} GB")
    if hw.gpus:
        for g in hw.gpus:
            print(f"  GPU {g.index}:    {g.name} ({g.vram_mb} MB)")
    else:
        print(f"  GPU:      None (CPU-only mode)")
    print()
