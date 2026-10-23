import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_model_package_has_no_rsi_code_dependency():
    for path in (ROOT / "src/deciform").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [entry.name for entry in node.names]
            else:
                continue
            assert not any(name.startswith(("jev_rsi", "typed_adapt")) for name in names), path


def test_model_training_entrypoints_import_independently():
    for name in ("train_episode", "evaluate_episode", "train_structured", "evaluate_adaptation"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)
