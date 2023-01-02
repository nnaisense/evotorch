# type: ignore
import os

import nox

DEFAULT_PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]
PYTHON_VERSIONS = os.environ.get("NOX_PYTHON_VERSIONS", ",".join(DEFAULT_PYTHON_VERSIONS)).split(",")
PYTEST_REPORT_FILE = os.environ.get("PYTEST_REPORT_FILE", "pytest-report.xml")


@nox.session(python=PYTHON_VERSIONS, venv_backend="conda", reuse_venv=True)
def tests_and_lint(session):
    session.install(".[dev]", silent=True)
    session.run("pytest", f"--junitxml={PYTEST_REPORT_FILE}")
    # session.run("mypy", ".", "--strict", silent=True)
    # session.run("isort", ".", "--check", silent=True)
    # session.run("black", "--check", ".", silent=True)
    # session.run("flake8")
