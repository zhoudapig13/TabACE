from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def check_module(name: str) -> dict[str, object]:
    try:
        module = __import__(name)
        return {"available": True, "version": getattr(module, "__version__", "unknown"), "error": ""}
    except Exception as exc:
        return {"available": False, "version": "", "error": repr(exc)}


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    results = {
        "tabicl": check_module("tabicl"),
        "tabpfn": check_module("tabpfn"),
        "evidence_boundary": (
            "Import availability does not make the main tables final TabICL/TabPFN model-level evidence. "
            "The main experiments isolate context retrieval under a local ICL-style fallback estimator; "
            "real backend smoke tests are reported separately."
        ),
    }
    output = ROOT / "results" / "backend_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
