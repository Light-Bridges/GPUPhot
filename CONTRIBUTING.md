# Contributing to GPUPhot

We welcome contributions to GPUPhot!  This document provides guidelines for contributing to the project. We appreciate
your help in making GPUPhot better.

## Getting Started

1. **Fork the Repository:** Click the "Fork" button in the top right corner of the GPUPhot GitHub
   repository ([https://github.com/Light-Bridges/GPUPhot](https://github.com/Light-Bridges/GPUPhot)) to create a copy of
   the repository in your own GitHub account.

2. **Clone Your Fork:** Clone your forked repository to your local machine:

   ```bash
   git clone https://github.com/<your-username>/GPUPhot.git  # Replace <your-username>
   cd GPUPhot
   ```

3. **Create a Virtual Environment (Recommended):**  Create and activate a Python virtual environment to isolate your
   development environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   .venv\\Scripts\\activate  # On Windows
   ```

4. **Install GPUPhot in Editable Mode:**  Install GPUPhot and its dependencies in "editable" mode:

   ```bash
   pip install -e .[dev]
   ```
   This installs GPUPhot in a way that any changes you make to the source code will be immediately reflected without
   needing to reinstall. The `[dev]` extra installs development dependencies (testing, documentation, etc.).

5. **Create a New Branch:** Create a new branch for your feature or bug fix, branching from the `main` branch:

   ```bash
   git checkout -b feature/my-awesome-feature  # For new features
   git checkout -b bugfix/fix-that-bug        # For bug fixes
   ```
   Use descriptive branch names.

## Making Changes

1. **Make Your Changes:**  Modify the code, add new features, or fix bugs.

2. **Add/Update Tests:**
    * Write unit tests for any new code you add.
    * Update existing tests if your changes affect existing functionality.
    * Tests are located in the `gpuphot/tests` directory.
    * We use `pytest` for testing.

3. **Run Tests:**  Make sure *all* tests pass before submitting your changes:

   ```bash
   pytest
   ```
   Or, if you have installed with the `[dev]` option:
   ```bash
    make test
   ```

4. **Update Documentation:**
    * If you add new features, update the documentation (including the README, docstrings, and any other relevant
      documentation files).
    * We use Sphinx to generate documentation. To build the documentation locally, you usually run `make html` from
      inside the `docs` directory. (You'll need to have Sphinx and the required extensions installed:
      `pip install -r requirements-docs.txt`).
    * Follow the Google style guide for docstrings. All public functions and classes should have docstrings.
    * Add type hints.

5. **Format Code:**  We use `black` as our code formatter to enforce a consistent code style. Run `black` before
   committing:

   ```bash
   black .
   ```
   You can also use:
   ```
   make format
   ```

6. **Lint Code:** We use `flake8` to check for style issues and potential errors. Run `flake8`:

   ```bash
   flake8
   ```
   Or:
   ```
    make lint
   ```
   Fix any issues reported by `flake8`.

7. **Commit with descriptive messages**:
   ```bash
   git commit -m "A concise and descriptive commit message"
   ```

## Submitting Changes

1. **Push to Your Fork:** Push your branch to your forked repository on GitHub:

   ```bash
   git push origin feature/my-awesome-feature
   ```

2. **Create a Pull Request:**
    * Go to the GPUPhot repository on
      GitHub: [https://github.com/Light-Bridges/GPUPhot](https://github.com/Light-Bridges/GPUPhot)
    * Click the "Pull Requests" tab.
    * Click the "New pull request" button.
    * Select your fork and branch as the "compare" branch, and the `main` branch of the main GPUPhot repository as the "
      base" branch.
    * Click "Create pull request".

3. **Pull Request Description:**
    * Provide a clear and concise description of your changes. Explain *what* you changed and *why*.
    * Reference any relevant issues (e.g., "Fixes #123").
    * If your changes are visual, include screenshots or GIFs.

4. **Review Process:**
    * Your pull request will be reviewed by the GPUPhot maintainers.
    * Be prepared to address any feedback or questions from the reviewers.
    * You may need to make further changes to your code based on the review.

5. **Continuous Integration:** Our CI system (GitHub Actions) will automatically run the tests and check the code style.
   Your pull request must pass all CI checks before it can be merged.

## Code Style

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
* Use `black` for code formatting.
* Use `flake8` for linting.
* Write clear and concise docstrings (Google style).
* Use type hints.

## Reporting Bugs

Use the GitHub issue tracker to report bugs. When reporting a bug, please include:

1. A clear and descriptive title.
2. A detailed description of the issue.
3. Steps to reproduce the problem (including the *exact* commands you ran, and the *exact* versions of the libraries you
   are using).
4. The expected behavior and the actual behavior.
5. Any relevant logs or error messages (use code blocks to format them).
6. Information about your environment (operating system, Python version, CUDA version, GPU model, etc.).

## Feature Requests

We welcome feature requests!  Please use the GitHub issue tracker to submit feature requests. Provide:

1. A clear and descriptive title.
2. A detailed description of the feature and its use case.
3. Examples of how the feature would be used.
4. If possible, suggest how the feature might be implemented.

## Questions

If you have any questions about contributing, please open an issue in the GitHub repository.

Thank you for contributing to GPUPhot!

## License

By contributing to GPUPhot, you agree that your contributions will be licensed under the [MIT License](LICENSE).
