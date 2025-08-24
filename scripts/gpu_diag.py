#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional


def run_cmd(cmd: List[str], timeout: int = 10) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        return out.strip()
    except Exception:
        return None


def gather_system_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PATH": os.environ.get("PATH"),
        },
    }


def gather_torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch  # type: ignore
    except Exception as e:
        info["error"] = f"torch import failed: {e}"
        return info

    info["torch_version"] = getattr(torch, "__version__", None)
    info["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    info["available"] = bool(torch.cuda.is_available())
    info["device_count"] = int(torch.cuda.device_count()) if info["available"] else 0

    devices: List[Dict[str, Any]] = []
    if info["available"]:
        for idx in range(info["device_count"]):
            props = torch.cuda.get_device_properties(idx)
            total, free = torch.cuda.mem_get_info(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "total_mem_mb": round(props.total_memory / (1024 ** 2), 2),
                    "mem_free_mb": round(free / (1024 ** 2), 2),
                    "mem_used_mb": round((total - free) / (1024 ** 2), 2),
                }
            )
        # live allocator stats (current process)
        try:
            devices_alloc: List[Dict[str, Any]] = []
            for idx in range(info["device_count"]):
                import torch  # re-import for type checkers
                torch.cuda.set_device(idx)
                devices_alloc.append(
                    {
                        "index": idx,
                        "allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024 ** 2), 2),
                        "reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024 ** 2), 2),
                        "max_allocated_mb": round(torch.cuda.max_memory_allocated(idx) / (1024 ** 2), 2),
                        "max_reserved_mb": round(torch.cuda.max_memory_reserved(idx) / (1024 ** 2), 2),
                    }
                )
            info["allocator"] = devices_alloc
        except Exception as e:
            info["allocator_error"] = str(e)

    info["devices"] = devices
    return info


def parse_nvidia_smi_query(output: str, fields: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            parts = [p.strip() for p in line.split(", ")]
        if len(parts) != len(fields):
            continue
        rows.append({f: v for f, v in zip(fields, parts)})
    return rows


def gather_nvidia_smi_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    smi = run_cmd(["nvidia-smi", "--version"])
    if not smi:
        return info
    info["available"] = True
    info["version_raw"] = smi

    gpu_fields = [
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "utilization.memory",
        "pstate",
        "temperature.gpu",
        "pcie.link.gen.current",
    ]
    q = run_cmd(["nvidia-smi", "--query-gpu=" + ",".join(gpu_fields), "--format=csv,noheader,nounits"])
    if q:
        info["gpus"] = parse_nvidia_smi_query(q, gpu_fields)

    proc_fields = ["pid", "process_name", "used_memory"]
    p = run_cmd(["nvidia-smi", "--query-compute-apps=" + ",".join(proc_fields), "--format=csv,noheader,nounits"])
    if p:
        info["processes"] = parse_nvidia_smi_query(p, proc_fields)
    else:
        p2 = run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
        if p2:
            info["processes"] = parse_nvidia_smi_query(p2, proc_fields)

    return info


def clean_torch_cuda_cache() -> Dict[str, Any]:
    result: Dict[str, Any] = {"attempted": False, "success": False}
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            result["attempted"] = True
            before = []
            for i in range(torch.cuda.device_count()):
                before.append(
                    {
                        "index": i,
                        "allocated_mb": round(torch.cuda.memory_allocated(i) / (1024 ** 2), 2),
                        "reserved_mb": round(torch.cuda.memory_reserved(i) / (1024 ** 2), 2),
                    }
                )
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            after = []
            for i in range(torch.cuda.device_count()):
                after.append(
                    {
                        "index": i,
                        "allocated_mb": round(torch.cuda.memory_allocated(i) / (1024 ** 2), 2),
                        "reserved_mb": round(torch.cuda.memory_reserved(i) / (1024 ** 2), 2),
                    }
                )
            result["before"] = before
            result["after"] = after
            result["success"] = True
        else:
            result["note"] = "CUDA not available"
    except Exception as e:
        result["error"] = str(e)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU/CUDA diagnostics and optional PyTorch cache cleanup")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--clean-cache", action="store_true", help="Run torch.cuda.empty_cache() and ipc_collect()")
    args = parser.parse_args()

    sys_info = gather_system_info()
    torch_info = gather_torch_info()
    smi_info = gather_nvidia_smi_info()
    clean_info = {}

    if args.clean_cache:
        clean_info = clean_torch_cuda_cache()
        torch_info_after = gather_torch_info()
    else:
        torch_info_after = {}

    if args.json:
        print(
            json.dumps(
                {
                    "system": sys_info,
                    "torch": torch_info,
                    "nvidia_smi": smi_info,
                    "clean": clean_info,
                    "torch_after_clean": torch_info_after,
                },
                indent=2,
            )
        )
        return 0

    print("System:")
    print(f"  platform: {sys_info['platform']}")
    print(f"  python  : {sys_info['python'].splitlines()[0]}")
    print(f"  CUDA_VISIBLE_DEVICES: {sys_info['env']['CUDA_VISIBLE_DEVICES']}\n")

    print("PyTorch:")
    print(f"  torch: {torch_info.get('torch_version')}, torch.cuda: {torch_info.get('torch_cuda_version')}, available: {torch_info.get('available')}")
    print(f"  device_count: {torch_info.get('device_count')}")
    for d in torch_info.get("devices", []):
        print(
            f"    [{d['index']}] {d['name']} (cc {d['capability']}) "
            f"total={d['total_mem_mb']}MB free={d['mem_free_mb']}MB used={d['mem_used_mb']}MB"
        )
    if "allocator" in torch_info:
        for a in torch_info["allocator"]:
            print(
                f"    alloc[{a['index']}]: allocated={a['allocated_mb']}MB reserved={a['reserved_mb']}MB "
                f"max_alloc={a['max_allocated_mb']}MB max_reserved={a['max_reserved_mb']}MB"
            )
    if err := torch_info.get("error"):
        print(f"  error: {err}")
    print()

    print("nvidia-smi:")
    if not smi_info.get("available"):
        print("  nvidia-smi not found or failed.")
    else:
        print(f"  version_raw: {smi_info.get('version_raw','').splitlines()[0]}")
        for g in smi_info.get("gpus", []):
            print(
                f"    {g['name']}: driver={g['driver_version']} mem={g['memory.used']}/{g['memory.total']} MB "
                f"util.gpu={g['utilization.gpu']}% util.mem={g['utilization.memory']}% pstate={g.get('pstate')} temp={g.get('temperature.gpu')}C"
            )
        procs = smi_info.get("processes", [])
        if procs:
            print("  processes:")
            for p in procs:
                print(f"    pid={p['pid']} name={p['process_name']} used_mem={p['used_memory']} MB")

    if args.clean_cache:
        print("\nCleanup (torch.cuda.empty_cache):")
        if clean_info.get("success"):
            for before, after in zip(clean_info.get("before", []), clean_info.get("after", [])):
                print(
                    f"  device[{before['index']}]: allocated {before['allocated_mb']}MB -> {after['allocated_mb']}MB, "
                    f"reserved {before['reserved_mb']}MB -> {after['reserved_mb']}MB"
                )
        else:
            print(f"  note/error: {clean_info.get('note') or clean_info.get('error')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


