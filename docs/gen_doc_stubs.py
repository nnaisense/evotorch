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

import mkdocs_gen_files
from rich import print

DEBUG = True


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

    if DEBUG:
        print(f"Generating docs for <{'.'.join(parts)}> -> '{full_doc_path}'")
    else:
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            print("::: " + ident, file=fd)

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

    return [".".join(parts[:2])] + parts[2:], doc_path


def generate_reference(src_dir: Path, target_dir: Path):
    """Generate the code reference pages and navigation."""
    nav = mkdocs_gen_files.Nav()
    for path in sorted(src_dir.glob("**/*.py")):
        key, path = generate_reference_for_file(path, src_dir=src_dir, target_dir=target_dir)
        if key:
            nav[key] = path

    if DEBUG:
        print(list(nav.build_literate_nav()))
    else:
        with mkdocs_gen_files.open(target_dir / "SUMMARY.md", "w") as nav_file:
            nav_file.writelines(nav.build_literate_nav())


generate_reference(Path("src"), Path("reference"))
