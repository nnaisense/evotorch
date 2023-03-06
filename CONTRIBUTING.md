# EvoTorch Contribution Guidelines

Thank you for your interest in contributing to EvoTorch!

We welcome any contributions to our Evotorch repository. Here are the guidelines for contributing to our project.

## Issue Tracker

If you find any bugs or issues with the project, please report them using the GitHub issue tracker.

## Getting started

1. We recommend that you start by creating an issue, explaining what contibution would you like to work on. This will ensure that you are working on something that is needed and will prevent duplication of effort. It also allows us to discuss and plan any implementation details.
2. Fork the repository on GitHub.
3. Clone the forked repository.
```bash
git clone <https://github.com/{your_username}/evotorch.git>
```
4. We would recommend creating a new conda environment from our `env.yaml` file, which will contain all development dependencies.
```bash
conda env create -f env.yaml
```
4. Alternativelly you can install all development dependencies using `pip`.
```bash
pip install -e .[dev]
```
5. Install `pre-commit` hooks. This will ensure you are using our code style.
```bash
pre-commit install
```

## Making changes

1. Make the necessary changes to the code.
2. Test your changes to ensure that everything is working properly. We are using `pytest` for writing and running tests.
3. Commit and push your changes
4. Create a **pull request** (PR) on GitHub to merge your changes into the main repository

## Code Guidelines

- Follow PEP8 + Black coding standards (`pre-commit` will help with that).
- Write clear and concise commit messages.
- Document your code using appropriate inline comments and clear docstrings. In case of a new bigger feature, feel free to recommend an extra doc page.
- Write unit tests for your code and ensure that they pass before submitting a pull request.
- When submitting PR, we expect clear summary of the purpose of the PR and the features it brings

While we appreciate the enthusiasm of any potential contributor and we are grateful for the PRs, we cannot guarantee to respond to or accept all PRs. In more details, we can:
- Reject a PR if we evaluate it as not suitable for the project's main repository
- Suggest to move a PR from the main EvoTorch repository to another EvoTorch-related repository

## Contact Us

If you have any questions or concerns about contributing to our project, you can also contact us at [evotorch@nnaisense.com](mailto:evotorch@nnaisense.com).

Thank you in advance, and looking forward to improving EvoTorch together! 🙌
