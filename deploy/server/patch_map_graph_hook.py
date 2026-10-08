"""Generate a reviewed coordinator candidate; never edit the server in place.

The caller must separately back up the live file and recheck its SHA256 before
deployment.  This patch accepts only the archived, inspected coordinator bytes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ORIGINAL_SHA256 = "c470e620fb3ce9487dfc6aa6989d95a6d0fd56cd9a4b349ce1d49a951a41e681"
MARKER = "# JAKA_GRAPH_UPDATE_HOOK_V1"
PUBLISH_ANCHOR = '''        atomic_publish_pair(
            publish_public,
            destination,
            publish_archive,
            archive_destination,
        )
'''
HOOK = '''        # JAKA_GRAPH_UPDATE_HOOK_V1
        # Graph failure is a degraded success: the validated map stays published.
        try:
            from map_graph_hook import update_published_graph
            manifest["graph_update"] = update_published_graph(
                destination,
                results_dir / "semantic_instance_graph.json",
                dataset_id=getattr(args, "graph_dataset_id", "jaka-mapping"),
                scene_id=getattr(args, "graph_scene_id", "scene-map"),
                run_id=run_id,
                every_nth_frame=int(args.every_nth_frame),
                skip=bool(getattr(args, "skip_graph_update", False)),
            )
        except Exception as graph_error:
            manifest["graph_update"] = {
                "status": "FAILED",
                "path": str(results_dir / "semantic_instance_graph.json"),
                "source_path": str(destination),
                "run_id": run_id,
                "error_type": type(graph_error).__name__,
                "error": str(graph_error) or type(graph_error).__name__,
            }
        if manifest["graph_update"]["status"] == "FAILED":
            print(
                "WARNING: scene map published, but entity graph update failed: "
                + manifest["graph_update"].get("error", "unknown error")
                + "; retry with map_graph_hook.py and the original run_id",
                file=sys.stderr,
            )
'''
ARG_ANCHOR = '''    parser.add_argument("--device", default="cuda")
'''
GRAPH_ARGS = '''    parser.add_argument("--graph-dataset-id", default="jaka-mapping")
    parser.add_argument("--graph-scene-id", default="scene-map")
    parser.add_argument("--skip-graph-update", action="store_true")
'''


def patch_bytes(original: bytes) -> bytes:
    digest = hashlib.sha256(original).hexdigest()
    if digest != ORIGINAL_SHA256:
        raise ValueError(f"unexpected coordinator SHA256: {digest}; expected {ORIGINAL_SHA256}")
    source = original.decode("utf-8")
    if MARKER in source:
        raise ValueError("coordinator already contains a graph update hook")
    for anchor in (PUBLISH_ANCHOR, ARG_ANCHOR):
        if source.count(anchor) != 1:
            raise ValueError("expected exactly one matching coordinator insertion point")
    candidate = source.replace(PUBLISH_ANCHOR, PUBLISH_ANCHOR + HOOK)
    candidate = candidate.replace(ARG_ANCHOR, GRAPH_ARGS + ARG_ANCHOR)
    ast.parse(candidate)
    return candidate.encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must be a new candidate file, not --input")
    original = args.input.read_bytes()
    try:
        candidate = patch_bytes(original)
    except ValueError as exc:
        parser.error(str(exc))
    # Exclusive creation prevents accidentally replacing an earlier candidate.
    with args.output.open("xb") as handle:
        handle.write(candidate)
    print(json.dumps({
        "input_sha256": hashlib.sha256(original).hexdigest(),
        "output_sha256": hashlib.sha256(candidate).hexdigest(),
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
