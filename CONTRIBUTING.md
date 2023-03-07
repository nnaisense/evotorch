# EvoTorch Contribution Guidelines

Thank you for your interest in contributing to EvoTorch!

Plans and/or ideas for contributing to EvoTorch are welcome! Below we list the contribution guidelines.

## Issue Tracker

If you find any bugs or issues with the project, please report them using the GitHub issue tracker.

## Getting started

We would like to recommend the following steps for getting started.

1. Create an issue, explaining what contribution you would like to work on. This will ensure that you are working on something that is needed and will prevent duplication of effort. It also allows us to discuss and plan any implementation details.
2. Fork the repository on GitHub.
3. Clone the forked repository.
```bash
git clone <https://github.com/{your_username}/evotorch.git>
```
4. We recommend creating a new conda environment from our `env.yaml` file, which will contain all development dependencies.
```bash
conda env create -f env.yaml
```
5. As an alternative to the `conda env create ...` command shown above, you could also create a new empty working environment (using `conda` or `virtualenv`), and then you could install into this new environment all development dependencies using `pip`:
```bash
pip install -e .[dev]
```
6. Install [pre-commit](https://pre-commit.com/) hooks. This will ensure you are using our code style.
```bash
pre-commit install
```

## Making changes

Please follow these steps for making changes:
1. Make the necessary changes to the code.
2. Test your changes to ensure that everything is working properly. We are using [pytest](https://pytest.org) for writing and running tests.
3. Commit and push your changes.
4. Create a **pull request** (PR) on GitHub to merge your changes into the main repository.

## Code Guidelines

We would be grateful if you could:
- Follow [PEP8](https://peps.python.org/pep-0008/) + [black](https://black.readthedocs.io/en/stable/index.html) coding standards ([pre-commit](https://pre-commit.com/) will help with that).
- Write clear and concise commit messages.
- Document your code using appropriate inline comments and clear docstrings. In case of a new bigger feature, feel free to recommend an extra doc page.
- Write unit tests for your code and ensure that they pass before submitting a pull request.
- When submitting the PR, provide a clear summary of the purpose of the PR and the features it brings.

While we appreciate the enthusiasm of any potential contributor and we are grateful for the PRs, we cannot guarantee to respond to or accept all PRs. In more details, we can:
- Reject a PR if we evaluate it as not suitable for the project's main repository
- Suggest to move a PR from the main EvoTorch repository to another EvoTorch-related repository

## Contact Us

If you have any questions or concerns about contributing to our project, you can also contact us at [evotorch@nnaisense.com](mailto:evotorch@nnaisense.com).

Thank you in advance, and looking forward to improving EvoTorch together! 🙌
