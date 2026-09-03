# -*- encoding: utf-8 -*-
#
# SWEaver: harmonic-domain manipulation of electomagnetic beams for CMB analysis
#
#           ##############
#        #######        #######
#      ####                  ####
#    ####                      ####
#   ###                          ###
#  ###  ####       ##       ####  ###
#  ## #######      ##      ####### ##
# ###########     ###      ###########
# ######  ###     ####     ### #######
# ####     ###    ####    ###     ####
# ###       ##   ######   ###       ##
#  ##       ###  ##  ##  ###       ##
#  ###      #######  #######      ###
#   ###      #####    #####      ###
#    ####    #####    #####    ####
#      ####                  ####
#        #######        #######
#            ##############
#
# Copyright © 2026 Maurizio Tomasi
# This code is licensed under the GPL 2
# See the file LICENSE.txt

from pathlib import Path
import importlib.resources


def get_test_data_path(identifier: str) -> Path:
    """
    Return the path to a .sph file used in SWEaver's tests.

    This function is useful for quickly getting sample data to test the library.
    Available identifiers typically correspond to the file names in the `tests/data`
    directory without the extension (e.g., 'hertzian_e_dipole_x', 'gaussian_beam').

    Args:
        identifier: A string identifying the test file (e.g. 'hertzian_e_dipole_x').
            It can also be the full file name (e.g. 'hertzian_e_dipole_x.sph').

    Returns:
        The absolute Path to the test file.

    Raises:
        FileNotFoundError: If the requested file cannot be found in the test data directory.
    """
    # Use importlib.resources to find the data files packaged inside the module
    data_dir = importlib.resources.files("sweaver") / "test_data"

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Directory {data_dir} does not exist. "
            "Ensure the test data is properly installed."
        )

    filename = identifier if ("." in identifier) else f"{identifier}.sph"

    candidate = data_dir / filename
    if candidate.is_file():
        # importlib.resources.abc.Traversable implements a subset of Path methods,
        # but in most cases for read operations casting to Path or string is preferred
        # but returning it directly as Path requires a path-like object or extracting it.
        # Since these are guaranteed to be local files, `as_posix()` or wrapping in `Path` works:
        return Path(str(candidate))

    # Try with .gz extension if not found
    candidate_gz = data_dir / f"{filename}.gz"
    if candidate_gz.is_file():
        return Path(str(candidate_gz))

    raise FileNotFoundError(
        f"Test data file for '{identifier}' not found in {data_dir}."
    )
