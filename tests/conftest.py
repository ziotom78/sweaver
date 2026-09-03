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
# This code is licensed under the GPL 3
# See the file LICENSE.txt

import pytest

from pathlib import Path
import numpy as np

from sweaver import ElectricField


@pytest.fixture
def data_dir() -> Path:
    """Return the absolute path to the directory containing test data."""
    return Path(__file__).parent / "data"


@pytest.fixture
def synthetic_efield():
    """
    Generates a synthetic ElectricField object representing a pure dipole.
    l=1, m=0.
    """
    lmax = 2
    mmax = 2

    # ElectricField expects an alm_stack of shape (4, nalm)
    # nalm for lmax=2, mmax=2 is ((2+1)*(2+2))//2 + (2+1)*(2-2) = 6
    nalm = ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)
    alm_stack = np.zeros((4, nalm), dtype=np.complex128)

    # Set a pure dipole mode (l=1, m=0) in the first component
    # Index for l=1, m=0 is 1
    idx_10 = 1
    alm_stack[0, idx_10] = 1.0 + 0j

    return ElectricField(frequency_ghz=100.0, lmax=lmax, mmax=mmax, alm_stack=alm_stack)
