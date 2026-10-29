"""OS process high-water marks (MiB), without polling or allocation tracing."""

from __future__ import annotations

import sys
from pathlib import Path

MIB = 1024 * 1024


def parse_linux_peak_rss(status: str) -> float:
    """Convert /proc/PID/status VmHWM from kernel kB (KiB) to MiB."""
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) != 3 or parts[2] != "kB" or int(parts[1]) <= 0:
                raise ValueError("Invalid Linux VmHWM field.")
            return int(parts[1]) / 1024
    raise RuntimeError("Linux /proc status does not expose VmHWM.")


def read_peak_rss() -> tuple[float, str]:
    """Read this process's lifetime resident-memory high-water mark.

    This is not exclusive algorithm memory. Call in a fresh process for each
    repetition; the counter cannot be reset between repetitions.
    """
    if sys.platform.startswith("linux"):
        return (
            parse_linux_peak_rss(Path("/proc/self/status").read_text(encoding="ascii")),
            "LinuxVmHWM",
        )
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if counters.PeakWorkingSetSize <= 0:
            raise RuntimeError("Windows returned an invalid peak working set.")
        return counters.PeakWorkingSetSize / MIB, "WindowsPeakWorkingSetSize"
    raise RuntimeError(
        "PeakRSS measurement currently supports Linux and Windows only. "
        "Run this experiment on Linux or Windows."
    )
