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


import subprocess as sp
from pathlib import Path
from typing import Optional

from mkdocs.structure.files import File

TMP_DIR = Path("/tmp").absolute()
NOTEBOOKS_DIR = Path(__file__).parent.parent / "examples" / "notebooks"
TARGET_DIR = TMP_DIR / "examples" / "notebooks"
GITHUB_BASE_ADDRESS = "https://github.com/nnaisense/evotorch/tree/master/examples/notebooks"
SKIP_NOTEBOOKS = [
    "reacher_train.ipynb",
]


def convert_notebook(notebook: Path, output_dir: Path) -> Optional[Path]:
    return_code = sp.call(["jupyter", "nbconvert", notebook, "--to", "markdown", "--output-dir", output_dir])
    if return_code == 0:
        return output_dir / notebook.with_suffix(".md").name
    else:
        return None


def add_link_to_github(md_file: Path, notebook: Path):
    rel_path = notebook.relative_to(NOTEBOOKS_DIR).with_suffix(".ipynb").as_posix()
    with open(md_file, "a") as f:
        f.write(
            "\n"
            "---\n"
            "\n"
            f"[See this notebook on GitHub]({GITHUB_BASE_ADDRESS}/{rel_path}){{ .md-button .md-button-primary }}",
        )


def convert_notebooks_in_dir(notebook_dir: Path):
    for notebook in notebook_dir.rglob("*.ipynb"):
        if notebook.name not in SKIP_NOTEBOOKS:
            md_file = convert_notebook(notebook, TARGET_DIR)
            if md_file:
                add_link_to_github(md_file, notebook)


def replace_in_file(fname: str, from_str: str, to_str: str):
    with open(fname, "r") as f:
        all_text = f.read()
    all_text = all_text.replace(from_str, to_str)
    with open(fname, "w") as f:
        f.write(all_text)


def fix_mpc_links():
    replace_in_file(
        TARGET_DIR / "reacher_mpc.md",
        "train_forward_model/reacher_train.ipynb",
        f"{GITHUB_BASE_ADDRESS}/Model_Predictive_Control_with_CEM/train_forward_model/reacher_train.ipynb",
    )


def convert_all_notebooks(**_):
    """Convert all notebooks to markdown files."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    convert_notebooks_in_dir(NOTEBOOKS_DIR)
    fix_mpc_links()


def append_notebooks(files, config):
    """Append notebooks to the list of files to be processed by mkdocs."""
    print("Appending notebooks to docs...")
    dest_dir = Path(files._files[0].abs_dest_path).parent
    for fname in TARGET_DIR.rglob("*.md"):
        file_name = (TARGET_DIR / fname).relative_to(TMP_DIR)
        f = File(path=file_name, src_dir=TMP_DIR.absolute(), dest_dir=dest_dir, use_directory_urls=True)
        # print(f"Appending file {f}")
        files.append(f)
