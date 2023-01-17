# Copyright 2022 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the code reference pages and navigation."""

from pathlib import Path
from typing import Optional, Union

import mkdocs_gen_files
from mkdocs.config import Config
from mkdocs.structure.files import File, Files
from mkdocs.structure.nav import Navigation
from nbconvert import MarkdownExporter

GITHUB_BASE_ADDRESS = "https://github.com/nnaisense/evotorch/tree/master/examples/notebooks"
SKIP_NOTEBOOKS = [
    "reacher_train.ipynb",
]


def convert_notebook(notebook_path: Path, doc_path: Path):
    md, res = MarkdownExporter().from_filename(notebook_path)
    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(md)


def add_link_to_github(doc_path: Path):
    rel_path = doc_path.with_suffix(".ipynb").as_posix()
    with mkdocs_gen_files.open(doc_path, "a") as f:
        f.write(
            "\n"
            "---\n"
            "\n"
            f"[See this notebook on GitHub]({GITHUB_BASE_ADDRESS}/{rel_path}){{ .md-button .md-button-primary }}",
        )


def convert_notebooks_in_dir(nav: mkdocs_gen_files.Nav, notebook_dir: Path, target_dir: Path):
    for notebook_path in notebook_dir.rglob("*.ipynb"):
        if notebook_path.name not in SKIP_NOTEBOOKS:

            notebook_rel_path = notebook_path.relative_to(notebook_dir).name
            doc_path = Path(target_dir, notebook_rel_path).with_suffix(".md")

            convert_notebook(notebook_path, doc_path)
            add_link_to_github(doc_path)
            nav[notebook_path.stem] = doc_path.name


def replace_in_file(fname: Path, from_str: str, to_str: str):
    with mkdocs_gen_files.open(fname, "r") as f:
        all_text = f.read()
    all_text = all_text.replace(from_str, to_str)
    with mkdocs_gen_files.open(fname, "w") as f:
        f.write(all_text)


def fix_mpc_links(target_dir: Path):
    replace_in_file(
        target_dir / "reacher_mpc.md",
        "train_forward_model/reacher_train.ipynb",
        f"{GITHUB_BASE_ADDRESS}/Model_Predictive_Control_with_CEM/train_forward_model/reacher_train.ipynb",
    )


def convert_all_notebooks(src_dir: Union[str, Path], target_dir: Union[str, Path]):
    """Convert all notebooks to markdown files."""
    if isinstance(src_dir, str):
        src_dir = Path(src_dir)

    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    nav = mkdocs_gen_files.Nav()

    convert_notebooks_in_dir(nav, src_dir, target_dir)
    fix_mpc_links(target_dir)

    # Note: Since we don't care about the order of the notebooks, we don't need to
    #       sort the navigation entries.
    # with mkdocs_gen_files.open(target_dir / "SUMMARY.md", "w") as nav_file:
    #     nav_file.writelines(nav.build_literate_nav())


def generate_reference_for_file(path: Path, src_dir: Path, target_dir: Path):
    # Ignore private modules
    if path.name.startswith("_") and not path.name == "__init__.py":
        return None, None

    doc_path = path.relative_to(src_dir).with_suffix(".md")
    full_doc_path = Path(target_dir, doc_path)

    # Split the module path into a list of directories and the module name
    parts = list(path.relative_to(src_dir).with_suffix("").parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    elif parts[-1] == "__main__":
        return None, None

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        print("::: " + ident, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

    return [".".join(parts[:2])] + parts[2:], doc_path


def generate_reference(src_dir: Union[str, Path], target_dir: Union[str, Path]):
    """Generate the code reference pages and navigation."""
    if isinstance(src_dir, str):
        src_dir = Path(src_dir)

    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    nav = mkdocs_gen_files.Nav()
    for path in sorted(src_dir.glob("**/*.py")):
        key, path = generate_reference_for_file(path, src_dir=src_dir, target_dir=target_dir)
        if key:
            nav[key] = path

    # with mkdocs_gen_files.open(target_dir / "SUMMARY.md", "w") as nav_file:
    #     nav_file.writelines(nav.build_literate_nav())


# Generate the code reference
generate_reference("src", "reference")

# Generate examples from notebooks
convert_all_notebooks("examples/notebooks", "examples/notebooks")
