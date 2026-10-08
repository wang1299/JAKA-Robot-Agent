"""Synchronize the entity graph after a public BoxFusion map is published.

Install beside ``update_map.py`` and ``sync_scene_graph.py``.  Snapshot here
means the entire *published JSON*, not proof that the physical environment
was completely observed.  Sampled batches therefore merge in partial mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


def _load_sync():
    # Import errors are contained by the hook: map publication stays successful.
    from sync_scene_graph import sync_graph
    return sync_graph


def update_published_graph(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dataset_id: str,
    scene_id: str,
    run_id: str,
    every_nth_frame: int | None = None,
    skip: bool = False,
) -> dict[str, Any]:
    """Return a manifest entry; graph errors never revoke a published map.

    ``sync_graph`` owns graph locking and atomic replacement.  This wrapper
    freezes the input bytes so another map publication cannot mix run IDs.
    """
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    result: dict[str, Any] = {
        "status": "SKIPPED" if skip else "FAILED",
        "path": str(output),
        "source_path": str(source),
        "source_sha256": None,
        "dataset_id": dataset_id,
        "scene_id": scene_id,
        "run_id": run_id,
        "revision": None,
        "counts": None,
        "scope": "published_map_snapshot_not_environment_coverage",
    }
    if skip:
        result["reason"] = "--skip-graph-update"
        return result
    try:
        if source == output:
            raise ValueError("graph output must not overwrite the published scene map")
        if not run_id or not dataset_id or not scene_id:
            raise ValueError("dataset_id, scene_id and run_id must be non-empty")
        raw = source.read_bytes()
        result["source_sha256"] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("published scene map must contain a JSON object")
        source_run = payload.get("run_id")
        if source_run is not None and source_run != run_id:
            raise ValueError("published map run_id differs from coordinator run_id")
        if every_nth_frame is None:
            every_nth_frame = (payload.get("sampling") or {}).get("every_nth_frame")
        if every_nth_frame is not None and (
            isinstance(every_nth_frame, bool)
            or not isinstance(every_nth_frame, int)
            or every_nth_frame < 1
        ):
            raise ValueError("every_nth_frame must be a positive integer")
        complete = every_nth_frame == 1
        result.update({
            "mode": "snapshot" if complete else "partial",
            "complete_snapshot": complete,
            "every_nth_frame": every_nth_frame,
        })
        sync_graph = _load_sync()
        with tempfile.TemporaryDirectory(prefix="jaka-graph-source-") as temporary:
            snapshot = Path(temporary) / "published_scene_map.json"
            snapshot.write_bytes(raw)
            summary = sync_graph(
                snapshot,
                output,
                dataset_id=dataset_id,
                scene_id=scene_id,
                run_id=run_id,
                mode=result["mode"],
                complete_snapshot=complete,
            )
        result.update({
            "status": "DONE",
            "revision": summary.get("revision"),
            "counts": summary.get("summary"),
            "input_counts": summary.get("input_counts"),
            "changed": summary.get("changed"),
            "idempotent": summary.get("idempotent"),
            "delta_counts": summary.get("delta_counts"),
        })
    except Exception as exc:
        result.update({
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc) or type(exc).__name__,
        })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="input_path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-id", default="jaka-mapping")
    parser.add_argument("--scene-id", default="scene-map")
    parser.add_argument("--run-id", required=True,
                        help="Original coordinator run_id; keep unchanged when retrying")
    parser.add_argument("--every-nth-frame", type=int, default=None,
                        help="Infer from map sampling metadata when omitted; unknown uses partial")
    args = parser.parse_args(argv)
    output = args.output or args.input_path.parent / "semantic_instance_graph.json"
    result = update_published_graph(
        args.input_path, output, dataset_id=args.dataset_id, scene_id=args.scene_id,
        run_id=args.run_id, every_nth_frame=args.every_nth_frame,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
