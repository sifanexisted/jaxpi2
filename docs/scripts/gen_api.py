"""Generate the API reference pages from jaxpi docstrings.

Run from the repo root:

    python docs/scripts/gen_api.py

One markdown page per module is written to docs/api/, with signatures,
docstrings, and source links to GitHub.
"""

import importlib
import inspect
import os
import textwrap

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "docs", "api")
GITHUB = "https://github.com/sifanexisted/jaxpi2/blob/main"

MODULES = [
    "models",
    "training",
    "archs",
    "samplers",
    "evaluator",
    "checkpointing",
    "logging",
    "utils",
]

SUMMARIES = {
    "models": "PINN base classes, model/optimizer factories, and the sharded training step.",
    "training": "Shared training loops: single runs and time-window curricula.",
    "archs": "Flax network architectures and input embeddings.",
    "samplers": "Infinite collocation-point samplers.",
    "evaluator": "Metric logging during training.",
    "checkpointing": "Orbax checkpointing and resume helpers.",
    "logging": "Tabulated console logging.",
    "utils": "Small utilities: pytree flattening, update schedules, schedule-free eval params.",
}


def source_link(obj):
    try:
        path = inspect.getsourcefile(obj)
        _, line = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return ""
    rel = os.path.relpath(path, REPO)
    if rel.startswith(".."):
        return ""
    return f" <a class='source-link' href='{GITHUB}/{rel}#L{line}' target='_blank'>[source]</a>"


def format_signature(name, obj):
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(...)"
    text = f"{name}{sig}"
    if len(text) > 88:  # break long signatures across lines
        try:
            params = ",\n    ".join(
                str(p) for p in inspect.signature(obj).parameters.values()
            )
            text = f"{name}(\n    {params},\n)"
        except (TypeError, ValueError):
            pass
    return text


def clean_doc(obj):
    doc = inspect.getdoc(obj)
    return textwrap.dedent(doc).strip() if doc else ""


def render_function(name, fn, level=3):
    lines = [f"{'#' * level} `{name}()`{source_link(fn)}", ""]
    lines += ["```python", format_signature(name, fn), "```", ""]
    doc = clean_doc(fn)
    if doc:
        lines += [doc, ""]
    return lines


def render_class(name, cls):
    lines = [f"## `{name}`{source_link(cls)}", ""]
    bases = [b.__name__ for b in cls.__bases__ if b is not object]
    if bases:
        lines += [f"*Bases: {', '.join(f'`{b}`' for b in bases)}*", ""]
    doc = clean_doc(cls)
    if doc:
        lines += [doc, ""]

    # public methods defined on this class (not inherited)
    for meth_name, meth in vars(cls).items():
        if meth_name.startswith("_") and meth_name != "__call__":
            continue
        fn = meth
        if isinstance(meth, (staticmethod, classmethod)):
            fn = meth.__func__
        if not callable(fn):
            continue
        display = f"{name}.{meth_name}"
        lines += render_function(display, fn, level=3)
    return lines


def main():
    os.makedirs(OUT, exist_ok=True)
    for mod_name in MODULES:
        module = importlib.import_module(f"jaxpi.{mod_name}")

        lines = [f"# jaxpi.{mod_name}", ""]
        summary = SUMMARIES.get(mod_name)
        mod_doc = clean_doc(module)
        if mod_doc:
            lines += [mod_doc, ""]
        elif summary:
            lines += [summary, ""]

        members = [
            (n, obj)
            for n, obj in vars(module).items()
            if not n.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
            and getattr(obj, "__module__", "") == f"jaxpi.{mod_name}"
        ]

        for n, obj in members:
            if inspect.isclass(obj):
                lines += render_class(n, obj)
            else:
                lines += render_function(n, obj, level=2)

        path = os.path.join(OUT, f"{mod_name}.md")
        with open(path, "w") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        print(f"docs/api/{mod_name}.md ({len(members)} members)")


if __name__ == "__main__":
    main()
