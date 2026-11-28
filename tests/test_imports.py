"""The package stands alone: src/ imports the standard library, this package and the listed libraries."""
import ast
import sys

import passing_check

# Widen this set on purpose, in the change that adds the dependency to pyproject.toml.
ALLOWED_THIRD_PARTY: frozenset[str] = frozenset({"numpy", "torch", "transformers"})


def test_repo_root_is_the_repo():
    assert (passing_check.REPO_ROOT / "pyproject.toml").exists()


def _top_level_imports(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_src_imports_only_the_standard_library_itself_and_the_listed_libraries():
    allowed = set(sys.stdlib_module_names) | {"passing_check"} | ALLOWED_THIRD_PARTY
    for p in sorted((passing_check.REPO_ROOT / "src").rglob("*.py")):
        extra = _top_level_imports(p.read_text(encoding="utf-8")) - allowed
        assert not extra, f"{p.name} imports {sorted(extra)}"


def test_the_import_check_catches_an_outside_package():
    assert _top_level_imports("import os\nfrom some_other_record import gate\n") - set(sys.stdlib_module_names) \
        == {"some_other_record"}
