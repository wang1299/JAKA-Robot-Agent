#!/usr/bin/env python3
"""Install/download/serve the two models locally; never connects to a GPU host."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
REPOSITORIES = {"qwen": "Qwen/Qwen3.5-9B", "minicpm": "openbmb/MiniCPM-V-4.6"}
NAMES = {"qwen": "Qwen3.5-9B", "minicpm": "MiniCPM-V-4.6"}


def read_config(path: Path | None, environ=None) -> dict[str, str]:
    """Parse plain KEY=VALUE, with no shell execution or arbitrary expansion."""
    result = {}
    if path is not None:
        for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            key, sep, value = raw.partition("=")
            key = key.strip()
            if not sep or not key.startswith("JAKA_") or not key.replace("_", "").isalnum():
                raise ValueError(f"Invalid config key on line {number}")
            parts = shlex.split(value, comments=True)
            if len(parts) != 1 or not parts[0]:
                raise ValueError(f"Expected one nonempty value on line {number}")
            result[key] = parts[0]
    result.update({k: v for k, v in (os.environ if environ is None else environ).items()
                   if k.startswith("JAKA_")})
    home = str(Path.home())
    return {k: v.replace("${HOME}", home).replace("$HOME", home) for k, v in result.items()}


@dataclass(frozen=True)
class Model:
    role: str
    weights: Path
    venv: Path
    gpu: str
    port: int
    dtype: str

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"


def settings(config: dict[str, str]) -> dict[str, Model]:
    home = Path(config.get("JAKA_MODEL_HOME", str(Path.home() / "jaka-models"))).expanduser()
    if not home.is_absolute():
        raise ValueError("JAKA_MODEL_HOME must be absolute")
    result = {}
    for role in REPOSITORIES:
        prefix = "JAKA_" + role.upper()
        weights = Path(config.get(prefix + "_MODEL", str(home / "models" / NAMES[role]))).expanduser()
        venv = Path(config.get(prefix + "_ENV", str(home / "envs" / role))).expanduser()
        if not weights.is_absolute() or not venv.is_absolute():
            raise ValueError("Model and environment directories must be absolute")
        port = int(config.get(prefix + "_PORT", "8001" if role == "qwen" else "8000"))
        if not 1 <= port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        result[role] = Model(role, weights, venv, config.get(prefix + "_GPU", "1" if role == "qwen" else "0"),
                             port, config.get("JAKA_MODEL_DTYPE", "bfloat16"))
    if result["qwen"].port == result["minicpm"].port:
        raise ValueError("The two services need different ports")
    if result["qwen"].venv == result["minicpm"].venv:
        raise ValueError("Use a separate environment for each model")
    if result["qwen"].weights == result["minicpm"].weights:
        raise ValueError("Use a separate weight directory for each model")
    return result


def serve_command(model: Model) -> list[str]:
    return [str(model.venv / "bin" / "transformers"), "serve", str(model.weights),
            "--host", "127.0.0.1", "--port", str(model.port), "--device", "cuda:0",
            "--dtype", model.dtype, "--attn-implementation", "sdpa",
            "--no-compile", "--no-continuous-batching", "--reasoning", "off"]


def install(model: Model, config: dict[str, str], python: str) -> None:
    if sys.platform != "linux":
        raise ValueError("GPU installation is supported on Linux; command/client-config work elsewhere")
    model.venv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([python, "-m", "venv", str(model.venv)], check=True)
    pip = [str(model.python), "-m", "pip"]
    subprocess.run(pip + ["install", "--upgrade", "pip"], check=True)
    torch = pip + ["install", "torch", "torchvision"]
    if config.get("JAKA_TORCH_INDEX_URL"):
        torch += ["--index-url", config["JAKA_TORCH_INDEX_URL"]]
    subprocess.run(torch, check=True)
    subprocess.run(pip + ["install", config.get("JAKA_TRANSFORMERS_SPEC", "transformers[serving]==5.12.1"),
                          "accelerate", "av", "pillow", "huggingface_hub"], check=True)
    subprocess.run(pip + ["check"], check=True)
    resolved = subprocess.check_output(pip + ["freeze"], text=True)
    (model.venv / "requirements.resolved.txt").write_text(resolved, encoding="utf-8")
    subprocess.run([str(model.python), "-c",
                    "import torch; print('PyTorch:', torch.__version__); "
                    "assert torch.cuda.is_available(), 'CUDA is unavailable; check driver and torch build'; "
                    "print('Visible GPUs:', torch.cuda.device_count())"], check=True)


def download(model: Model, revision: str | None) -> None:
    from huggingface_hub import HfApi, snapshot_download
    repository = REPOSITORIES[model.role]
    sha = HfApi().model_info(repository, revision=revision or "main").sha
    if not sha:
        raise RuntimeError("Could not resolve the model revision")
    snapshot_download(repository, revision=sha, local_dir=str(model.weights))
    manifest = {"repository": repository, "revision": sha, "local_dir": str(model.weights)}
    (model.weights / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {repository}; revision recorded in download-manifest.json")


def client_config(models: dict[str, Model]) -> dict[str, str]:
    return {"JAKA_AGENT_PROTOCOL": "json",
            "JAKA_AGENT_BASE_URL": f"http://127.0.0.1:{models['qwen'].port}/v1",
            "JAKA_AGENT_MODEL": str(models["qwen"].weights),
            "DASHSCOPE_BASE_URL": f"http://127.0.0.1:{models['minicpm'].port}/v1",
            "QWEN_PLAN_MODEL": str(models["minicpm"].weights),
            "QWEN_VISION_MODEL": str(models["minicpm"].weights)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "download", "serve", "command", "client-config"])
    parser.add_argument("--role", choices=["all", "qwen", "minicpm"], default="all")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--python", default="python3", help="Linux Python used to create GPU environments")
    parser.add_argument("--revision", help="Hugging Face commit/tag; download one role at a time")
    parser.add_argument("--output", type=Path, help="New client JSON; refuses to overwrite existing files")
    args = parser.parse_args(argv)
    default = REPO / "configs" / "server.local.env"
    path = args.config if args.config is not None else (default if default.exists() else None)
    try:
        config = read_config(path)
        models = settings(config)
        if args.action in {"serve", "command"} and args.role == "all":
            parser.error("serve/command need --role qwen or --role minicpm")
        if args.revision and (args.action != "download" or args.role == "all"):
            parser.error("--revision needs download and one --role")
        if args.action == "client-config":
            if args.output is None:
                parser.error("client-config needs --output")
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(client_config(models), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print("Client config created; copy it privately to the robot")
            return 0
        for role in (list(models) if args.role == "all" else [args.role]):
            model = models[role]
            if args.action == "install":
                install(model, config, args.python)
            elif args.action == "download":
                download(model, args.revision)
            elif args.action == "command":
                print("CUDA_VISIBLE_DEVICES=" + shlex.quote(model.gpu) + " " + shlex.join(serve_command(model)))
            else:
                if not model.weights.joinpath("config.json").is_file():
                    raise ValueError(f"Missing config.json in weight directory for {role}; download first")
                if not model.venv.joinpath("bin", "transformers").is_file():
                    raise ValueError(f"Missing transformers executable for {role}; install first")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=model.gpu,
                           HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
                command = serve_command(model)
                os.execvpe(command[0], command, env)
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Deployment failed ({type(exc).__name__}); check local paths, dependencies and config", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
