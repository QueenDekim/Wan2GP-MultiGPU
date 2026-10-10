#!/usr/bin/env python3
"""Static GPU-first compatibility inventory for Wan2GP model integrations.

No model imports, checkpoint downloads, CUDA drivers or external packages are
needed. This is an *audit*, not a substitute for real GPU inference tests.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path


MODEL_METHODS = frozenset({"forward", "generate", "inference", "encode", "decode"})
HEAVY_RUNTIME_DIRS = frozenset({"models", "shared", "preprocessing", "postprocessing"})


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return (prefix + "." if prefix else "") + node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _cpu_argument(call: ast.Call) -> bool:
    args = list(call.args)
    args += [kw.value for kw in call.keywords if kw.arg == "device"]
    return any(isinstance(v, ast.Constant) and str(v.value).lower() == "cpu" for v in args)


def inspect_file(path: Path, root: Path) -> dict:
    relative = path.relative_to(root).as_posix()
    try:
        data = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(data, filename=relative)
    except (OSError, UnicodeError, SyntaxError) as exc:
        return {"file": relative, "parse_error": str(exc)}

    families = []
    parts = path.relative_to(root).parts
    if len(parts) >= 2 and parts[0] == "models":
        families.append(parts[1])

    findings = []
    counters = Counter()
    classes = set()

    class Walker(ast.NodeVisitor):
        def __init__(self):
            self.class_name = None
            self.method_name = None

        def visit_ClassDef(self, node):
            old = self.class_name
            self.class_name = node.name
            classes.add(node.name)
            self.generic_visit(node)
            self.class_name = old

        def visit_FunctionDef(self, node):
            old = self.method_name
            self.method_name = node.name
            if self.class_name and node.name == "to":
                positional = [arg.arg for arg in node.args.posonlyargs + node.args.args]
                kwonly = [arg.arg for arg in node.args.kwonlyargs]
                if "device" in kwonly and "device" not in positional:
                    findings.append({
                        "line": node.lineno,
                        "risk": "custom_to_keyword_only",
                        "owner": self.class_name,
                        "detail": "Accelerate may invoke .to(device, non_blocking=True)",
                    })
            self.generic_visit(node)
            self.method_name = old

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            name = _call_name(node.func)
            if not name:
                self.generic_visit(node)
                return
            if name.endswith(".to") and _cpu_argument(node):
                counters["explicit_cpu_transfer"] += 1
                if self.method_name in MODEL_METHODS or self.method_name == "release":
                    findings.append({
                        "line": node.lineno,
                        "risk": "cpu_transfer_in_" + self.method_name,
                        "owner": self.class_name,
                        "detail": name,
                    })
            elif name.endswith(".cpu"):
                counters["explicit_cpu_transfer"] += 1
            elif name.endswith(".cuda"):
                counters["hardcoded_cuda"] += 1
            elif name in ("torch.cuda.empty_cache", "torch.cuda.ipc_collect"):
                counters["cuda_cache_operations"] += 1
                if self.method_name in MODEL_METHODS:
                    findings.append({
                        "line": node.lineno,
                        "risk": "cache_operation_in_hot_path",
                        "owner": self.class_name,
                        "detail": name,
                    })
            if name in ("torch.from_numpy", "torch.as_tensor"):
                counters["numpy_tensor_boundary"] += 1
            self.generic_visit(node)

        def visit_Name(self, node):
            if "lora" in node.id.lower():
                counters["lora_references"] += 1

    Walker().visit(tree)
    return {
        "file": relative,
        "families": families,
        "classes": sorted(classes),
        "counters": dict(counters),
        "findings": findings,
    }


def audit(root: Path) -> dict:
    roots = [root / name for name in HEAVY_RUNTIME_DIRS if (root / name).exists()]
    results = [
        inspect_file(path, root)
        for folder in roots
        for path in sorted(folder.rglob("*.py"))
        if not any(part in ("__pycache__", ".venv", "venv") for part in path.parts)
    ]
    totals = Counter()
    families = defaultdict(Counter)
    risks = Counter()
    findings = []
    parse_errors = []
    for result in results:
        if "parse_error" in result:
            parse_errors.append({"file": result["file"], "error": result["parse_error"]})
            continue
        totals.update(result["counters"])
        for family in result["families"]:
            families[family]["files"] += 1
            families[family].update(result["counters"])
        for finding in result["findings"]:
            entry = {"file": result["file"], **finding}
            findings.append(entry)
            risks[finding["risk"]] += 1

    return {
        "python_files": len(results),
        "model_families": dict(sorted((k, dict(v)) for k, v in families.items())),
        "counters": dict(totals),
        "risks": dict(risks),
        "findings": findings,
        "parse_errors": parse_errors,
        "notes": [
            "Findings require manual review: a CPU tensor may be required at a file I/O boundary.",
            "Explicit model sharding covers known architectures; inferred block sharding is conservative.",
            "The audit cannot establish numerical correctness, VRAM peaks or inference latency.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(
        f"[MultiGPU audit] Scanned {report['python_files']} Python files, "
        f"{len(report['model_families'])} model families; "
        f"{len(report['findings'])} review findings; "
        f"{len(report['parse_errors'])} parse errors"
    )
    print("Risks:", json.dumps(report["risks"], sort_keys=True))
    if report["parse_errors"]:
        print("Parse errors:", json.dumps(report["parse_errors"][:10]))


if __name__ == "__main__":
    main()
