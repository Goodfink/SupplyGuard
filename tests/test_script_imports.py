import importlib
import subprocess
import sys


def test_importing_scripts_has_no_side_effects(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))

    for module_name in ("src.rag_cli", "src.run_baseline", "src.run_experiments"):
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)

    assert calls == []
