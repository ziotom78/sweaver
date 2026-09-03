import nox

nox.options.default_venv_backend = "uv"


# Define the versions you want to test against
@nox.session(python=["3.11", "3.14"])
def tests(session):
    """Run the test suite."""
    session.run("uv", "sync", "--python", session.python, external=True)
    session.run("uv", "run", "python3", "-m", "pytest", external=True)


@nox.session(python="3.11")
def test_install(session):
    """
    Test that the package can be built, installed, and that data files are accessible.

    This ensures that data files are correctly packaged inside the wheel
    and available to end-users after pip install.
    """
    import glob
    import os

    # 1. Build the wheel package in the 'dist/' directory
    session.run("uv", "build", "--wheel", external=True)

    # 2. Find the newly built wheel file.
    # We must resolve the absolute path *before* changing the directory!
    wheels = glob.glob("dist/*.whl")
    if not wheels:
        session.error("No wheel found in dist/")
    latest_wheel = os.path.abspath(max(wheels, key=os.path.getmtime))

    # 3. Create a temporary directory and switch to it
    # This prevents Python from importing the local 'src/sweaver'
    # instead of the installed wheel.
    tmp_dir = session.create_tmp()
    session.chdir(tmp_dir)

    # 4. Install the wheel into the Nox virtual environment
    # Since we use 'uv' as venv backend, we can use it to install
    session.run("uv", "pip", "install", latest_wheel, external=True)

    # 5. Run a quick script to test that 'importlib.resources' successfully finds the data
    script = (
        "import sweaver; "
        "path = sweaver.get_test_data_path('hertzian_e_dipole_x'); "
        "assert path.exists(), f'File not found: {path}'; "
        "print(f'Test data successfully found at: {path}')"
    )
    session.run("python", "-c", script)


@nox.session
def docs(session):
    """Build the documentation using Sphinx."""
    session.run("uv", "sync", "--group", "docs", external=True)
    session.run(
        "uv",
        "run",
        "sphinx-build",
        "-b",
        "html",
        "docs/",
        "docs/_build/html",
        external=True,
    )
