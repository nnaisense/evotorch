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


import os
import subprocess as sp
from typing import Optional

DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
NOTEBOOKS_DIR = os.path.abspath(os.path.join(DOCS_DIR, "..", "examples", "notebooks"))
TARGET_DIR = os.path.join(DOCS_DIR, "examples", "notebooks")
GITHUB_BASE_ADDRESS = "https://github.com/nnaisense/evotorch/tree/master/examples/notebooks"


ALL_NOTEBOOK_DIRS = [
    (NOTEBOOKS_DIR, GITHUB_BASE_ADDRESS),
    (
        os.path.join(NOTEBOOKS_DIR, "Model_Predictive_Control_with_CEM"),
        GITHUB_BASE_ADDRESS + "/Model_Predictive_Control_with_CEM",
    ),
]


def convert_notebook(fname: str, output_dir: str) -> Optional[str]:
    _, notebook_name = os.path.split(fname)

    if notebook_name.lower().endswith(".ipynb"):
        raw_name = notebook_name[:-6]
    else:
        raise ValueError(f"Invalid notebook name: {notebook_name}")

    md_name = raw_name + ".md"

    return_code = sp.call(["jupyter", "nbconvert", fname, "--to", "markdown", "--output-dir", output_dir])
    if return_code == 0:
        return os.path.join(output_dir, md_name)
    else:
        return None


def add_lines_to_text_file(fname: str, lines: list):
    with open(fname, "a") as f:
        for line in lines:
            print(line, file=f)


def convert_notebooks_in_dir(notebook_dir: str, github_dir: str):
    for fname in os.listdir(notebook_dir):
        if fname.lower().endswith(".ipynb"):
            full_fname = os.path.join(notebook_dir, fname)
            full_md_name = convert_notebook(full_fname, TARGET_DIR)
            if full_md_name is not None:
                github_address = github_dir + "/" + fname
                add_lines_to_text_file(
                    full_md_name,
                    [
                        "",
                        "---",
                        "",
                        f"[See this notebook on GitHub]({github_address}){{ .md-button .md-button-primary }}",
                    ],
                )


def replace_in_file(fname: str, from_str: str, to_str: str):
    with open(fname, "r") as f:
        all_text = f.read()
    all_text = all_text.replace(from_str, to_str)
    with open(fname, "w") as f:
        f.write(all_text)


def fix_mpc_links():
    replace_in_file(
        os.path.join(TARGET_DIR, "reacher_mpc.md"),
        "train_forward_model/reacher_train.ipynb",
        f"{GITHUB_BASE_ADDRESS}/Model_Predictive_Control_with_CEM/train_forward_model/reacher_train.ipynb",
    )


def convert_all_notebooks(*args, **kwargs):
    os.makedirs(TARGET_DIR, exist_ok=True)
    for notebook_dir, github_dir in ALL_NOTEBOOK_DIRS:
        convert_notebooks_in_dir(notebook_dir, github_dir)
    fix_mpc_links()
