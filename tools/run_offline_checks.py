"""Run portable unit and browser-contract checks without model or robot access."""
import argparse
import contextlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MANUAL = {"test_service.py", "test_reference_match.py", "test_minicpm_reference_exact.py"}
HISTORY_ONLY = {"test_original_find_contract.py"}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def run_python(output):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    files = [p for p in sorted((ROOT / "tests").glob("test_*.py"))
             if p.name not in MANUAL | HISTORY_ONLY]
    original_connect = socket.socket.connect

    def loopback_only(sock, address):
        if isinstance(address, tuple):
            host = address[0]
            try:
                allowed = host == "localhost" or ipaddress.ip_address(host).is_loopback
            except ValueError:
                allowed = False
            if not allowed:
                raise RuntimeError("Offline checks block non-loopback connections")
        return original_connect(sock, address)

    with (output / "python.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), \
                patch.object(socket.socket, "connect", loopback_only):
            tests = {}
            for path in files:
                for test in flatten(unittest.defaultTestLoader.loadTestsFromName("tests." + path.stem)):
                    tests.setdefault(test.id(), test)
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(
                unittest.TestSuite(tests.values()))
    return {
        "modules": len(files), "tests_run": result.testsRun,
        "failures": [{"test": t.id(), "traceback": detail} for t, detail in result.failures],
        "errors": [{"test": t.id(), "traceback": detail} for t, detail in result.errors],
        "skipped": [{"test": t.id(), "reason": reason} for t, reason in result.skipped],
        "successful": result.wasSuccessful(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", help="Node.js executable; otherwise search PATH")
    parser.add_argument("--require-node", action="store_true", help="Fail if Node.js is unavailable")
    args = parser.parse_args()
    os.chdir(ROOT)
    output = ROOT / "tests/artifacts/offline-checks"
    output.mkdir(parents=True, exist_ok=True)
    python_result = run_python(output)
    print(f"Python: {python_result['tests_run']} tests; "
          f"{len(python_result['failures'])} failures, {len(python_result['errors'])} errors, "
          f"{len(python_result['skipped'])} skipped")
    node = args.node or shutil.which("node")
    frontend = []
    ok = python_result["successful"]
    if node:
        for script in sorted((ROOT / "tests").glob("*.cjs")):
            try:
                result = subprocess.run([node, str(script)], cwd=ROOT, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace", timeout=60)
                detail = result.stdout + result.stderr
                passed = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired) as exc:
                detail, passed = str(exc), False
            (output / (script.stem + ".log")).write_text(detail, encoding="utf-8")
            frontend.append({"script": script.name, "successful": passed})
            print(f"Frontend: {script.name}: {'PASS' if passed else 'FAIL'}")
            ok = ok and passed
    else:
        print("Frontend: SKIPPED (Node.js not found)")
        if args.require_node:
            ok = False
    report = {"python": python_result, "frontend": frontend,
              "frontend_skipped": not bool(node), "excluded_manual_scripts": sorted(MANUAL),
              "non_loopback_python_connections_blocked": True, "successful": ok}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"Details: {output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
