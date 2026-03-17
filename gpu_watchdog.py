#!/usr/bin/env python3
"""
GPU Inactivity Watchdog

Polls RunPod API for GPU utilization from the local Mac. Stops pods that have
been idle longer than --timeout minutes. No per-pod setup required.

Usage:
    uv run --no-project --env-file .env python gpu_watchdog.py --pods <id> [<id> ...]
    uv run --no-project --env-file .env python gpu_watchdog.py --pods abc123 --timeout 2 --interval 10

.env needs: RUNPOD_API_KEY=your_key
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

QUERY_POD = """
query GetPod($podId: String!) {
  pod(input: { podId: $podId }) {
    id
    name
    desiredStatus
    runtime {
      gpus {
        gpuUtilPercent
        memoryUtilPercent
      }
    }
  }
}
"""

MUTATION_STOP = """
mutation StopPod($podId: String!) {
  podStop(input: { podId: $podId }) {
    id
  }
}
"""

# ─── .env loader ──────────────────────────────────────────────────────────────


def load_env(path: str = ".env") -> None:
    """Parse KEY=VALUE lines from a .env file into os.environ."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ─── Logging ──────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── RunPod API ───────────────────────────────────────────────────────────────


def runpod_request(api_key: str, query: str, variables: dict) -> dict | None:
    """Execute a GraphQL query/mutation against the RunPod API."""
    url = f"{RUNPOD_GRAPHQL_URL}?api_key={api_key}"
    payload = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log(f"WARNING: API request failed — {e}")
        return None


def get_pod_status(api_key: str, pod_id: str) -> dict | None:
    """Return pod dict with id, name, desiredStatus, runtime.gpus, or None on failure."""
    result = runpod_request(api_key, QUERY_POD, {"podId": pod_id})
    if result is None:
        return None
    errors = result.get("errors")
    if errors:
        log(f"WARNING: API errors for pod {pod_id}: {errors}")
        return None
    return result.get("data", {}).get("pod")


def stop_pod(api_key: str, pod_id: str) -> bool:
    """Send podStop mutation. Returns True if the response confirms the stop."""
    result = runpod_request(api_key, MUTATION_STOP, {"podId": pod_id})
    if result is None:
        return False
    errors = result.get("errors")
    if errors:
        log(f"WARNING: podStop errors for {pod_id}: {errors}")
        return False
    stopped_id = result.get("data", {}).get("podStop", {}).get("id")
    return stopped_id is not None


# ─── Watchdog loop ────────────────────────────────────────────────────────────


def watchdog_loop(
    api_key: str,
    pod_ids: list[str],
    timeout_secs: int,
    threshold_pct: float,
    interval_secs: int,
) -> None:
    # idle_seconds[pod_id] = accumulated idle seconds; None means "not tracked yet"
    idle_seconds: dict[str, int] = {pod_id: 0 for pod_id in pod_ids}
    remaining = set(pod_ids)

    log(f"Watching {len(pod_ids)} pod(s): {', '.join(pod_ids)}")
    log(f"Idle threshold: {threshold_pct}% GPU util | Timeout: {timeout_secs}s | Poll: {interval_secs}s")

    while remaining:
        time.sleep(interval_secs)

        for pod_id in list(remaining):
            pod = get_pod_status(api_key, pod_id)

            if pod is None:
                log(f"[{pod_id}] API failure — skipping poll (not accumulating idle time)")
                continue

            desired = pod.get("desiredStatus", "")
            name = pod.get("name", pod_id)

            if desired != "RUNNING":
                log(f"[{pod_id}] ({name}) desiredStatus={desired!r} — already stopped, removing from watch")
                remaining.discard(pod_id)
                continue

            runtime = pod.get("runtime")
            if not runtime:
                log(f"[{pod_id}] ({name}) no runtime info yet — skipping poll")
                continue

            gpus = runtime.get("gpus") or []
            if not gpus:
                log(f"[{pod_id}] ({name}) no GPU data — skipping poll")
                continue

            max_util = max((g.get("gpuUtilPercent", 0) or 0) for g in gpus)

            if max_util >= threshold_pct:
                idle_seconds[pod_id] = 0
                log(f"[{pod_id}] ({name}) GPU active — util={max_util:.1f}% (idle timer reset)")
            else:
                idle_seconds[pod_id] += interval_secs
                idle_mins = idle_seconds[pod_id] / 60
                log(
                    f"[{pod_id}] ({name}) GPU idle — util={max_util:.1f}% "
                    f"(idle {idle_mins:.1f}/{timeout_secs / 60:.1f} min)"
                )

                if idle_seconds[pod_id] >= timeout_secs:
                    log(f"[{pod_id}] ({name}) IDLE TIMEOUT reached — sending podStop...")
                    ok = stop_pod(api_key, pod_id)
                    if ok:
                        log(f"[{pod_id}] ({name}) Pod stopped successfully.")
                    else:
                        log(f"[{pod_id}] ({name}) WARNING: podStop may have failed — check console.")
                    remaining.discard(pod_id)

    log("All monitored pods have been stopped. Exiting.")


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU inactivity watchdog — stops idle RunPod pods from the local Mac."
    )
    parser.add_argument(
        "--pods",
        nargs="+",
        required=True,
        metavar="POD_ID",
        help="One or more RunPod pod IDs to monitor",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        metavar="MINUTES",
        help="Minutes of GPU idle before stopping the pod (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        metavar="PCT",
        help="GPU utilization %% below which the pod is considered idle (default: 5)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Seconds between polls (default: 60)",
    )
    return parser.parse_args()


def main() -> None:
    load_env(".env")
    args = parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        log("ERROR: RUNPOD_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    timeout_secs = int(args.timeout * 60)
    interval_secs = int(args.interval)

    watchdog_loop(
        api_key=api_key,
        pod_ids=args.pods,
        timeout_secs=timeout_secs,
        threshold_pct=args.threshold,
        interval_secs=interval_secs,
    )


if __name__ == "__main__":
    main()
