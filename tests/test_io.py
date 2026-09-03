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

import gzip
import sweaver

from pathlib import Path
import numpy as np
import numpy.testing as npt

import pytest

DATA_DIR = Path(__file__).parent / "data"
SQRT_2 = np.sqrt(2)


@pytest.mark.parametrize(
    "data_file,reference",
    [
        (
            "hertzian_e_dipole_x.sph",
            {(2, -1, 1): -1j / SQRT_2, (2, 1, 1): 1j / SQRT_2},
        ),
        ("hertzian_e_dipole_y.sph", {(2, -1, 1): 1 / SQRT_2, (2, 1, 1): 1 / SQRT_2}),
        ("hertzian_e_dipole_z.sph", {(2, 0, 1): -1j}),
        ("hertzian_h_dipole_x.sph", {(1, -1, 1): 1 / SQRT_2, (1, 1, 1): -1 / SQRT_2}),
        (
            "hertzian_h_dipole_y.sph",
            {(1, -1, 1): 1j / SQRT_2, (1, 1, 1): 1j / SQRT_2},
        ),
        ("hertzian_h_dipole_z.sph", {(1, 0, 1): 1}),
    ],
)
def test_hertzian_dipoles(data_file, reference):
    with (sweaver.get_test_data_path(data_file)).open("rt") as f:
        grasp_file = sweaver.read_sph_file(f)

    assert grasp_file.num_of_blocks == 1
    freq_block = grasp_file.get(index=0)

    npt.assert_allclose(freq_block.cum_power, 0.5)

    nmax = freq_block.header.nmax
    mmax = freq_block.header.mmax
    for s in (1, 2):
        for n in range(1, nmax + 1):
            for m in range(-mmax, mmax + 1):
                idx = (s, m, n)
                if idx in reference.keys():
                    npt.assert_allclose(
                        freq_block.get_q(*idx).real, reference[idx].real, atol=1e-12
                    )
                    npt.assert_allclose(
                        freq_block.get_q(*idx).imag, reference[idx].imag, atol=1e-12
                    )
                else:
                    npt.assert_allclose(freq_block.get_q(*idx).real, 0.0, atol=1e-12)
                    npt.assert_allclose(freq_block.get_q(*idx).imag, 0.0, atol=1e-12)


def test_read_multi_frequency():
    with (sweaver.get_test_data_path("multi_frequency.sph")).open("rt") as f:
        grasp_file = sweaver.read_sph_file(f)

    assert grasp_file.num_of_blocks == 2
    assert grasp_file.get(index=0).frequency_ghz == 15.0
    assert grasp_file.get(index=1).frequency_ghz == 17.0


def test_read_sph_frequency():
    # Test reading from a GZipped file path
    gz_path = sweaver.get_test_data_path("gaussian_beam.sph.gz")
    freq_block_gz = sweaver.read_sph_frequency_block(gz_path)
    assert freq_block_gz.frequency_ghz == pytest.approx(15.0)
    npt.assert_allclose(freq_block_gz.cum_power, 0.5, atol=1e-6)

    # Test reading from a plain text file path
    plain_path = sweaver.get_test_data_path("hertzian_e_dipole_x.sph")
    freq_block_plain = sweaver.read_sph_frequency_block(plain_path)
    assert freq_block_plain.frequency_ghz == pytest.approx(15.0)
    npt.assert_allclose(freq_block_plain.cum_power, 0.5, atol=1e-6)

    # Test reading from a file-like object (uncompressed)
    with plain_path.open("rt") as f:
        freq_block_fobj = sweaver.read_sph_frequency_block(f)
        assert freq_block_fobj.frequency_ghz == pytest.approx(15.0)
        npt.assert_allclose(freq_block_fobj.cum_power, 0.5, atol=1e-6)

    # Test reading from a file-like object (GZipped)
    with gzip.open(gz_path, "rt") as f:
        freq_block_fobj_gz = sweaver.read_sph_frequency_block(f)
        assert freq_block_fobj_gz.frequency_ghz == pytest.approx(15.0)
        npt.assert_allclose(freq_block_fobj_gz.cum_power, 0.5, atol=1e-6)


def test_convert_to_electric_field():

    def get_coeffs(file_name: str, ell: int, m: int):
        with (sweaver.get_test_data_path(file_name)).open("rt") as f:
            grasp_file = sweaver.read_sph_file(f)

        assert grasp_file.num_of_blocks == 1
        electric_field = sweaver.ElectricField.from_frequency_block(
            grasp_file.get(index=0)
        )

        e_re, b_re, e_im, b_im = electric_field.get_alms(ell=ell, m=m)
        scale = np.sqrt(4 * np.pi)
        return e_re / scale, b_re / scale, e_im / scale, b_im / scale

    sqrt2 = np.sqrt(2)

    # Electric dipole along X
    e_re, b_re, e_im, b_im = get_coeffs("hertzian_e_dipole_x.sph", 1, 1)

    npt.assert_allclose(e_re, 1 / sqrt2 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_re, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(e_im, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_im, 0.0 + 0.0j, atol=1e-15)

    # Electric dipole along Y
    e_re, b_re, e_im, b_im = get_coeffs("hertzian_e_dipole_y.sph", 1, 1)

    npt.assert_allclose(e_re, -1j / sqrt2, atol=1e-15)
    npt.assert_allclose(b_re, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(e_im, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_im, 0.0 + 0.0j, atol=1e-15)

    # Electric dipole along Z
    e_re, b_re, e_im, b_im = get_coeffs("hertzian_e_dipole_z.sph", 1, 0)

    npt.assert_allclose(e_re, -1.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_re, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(e_im, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_im, 0.0 + 0.0j, atol=1e-15)

    # Magnetic dipole along X
    e_re, b_re, e_im, b_im = get_coeffs("hertzian_h_dipole_x.sph", 1, 1)

    npt.assert_allclose(e_re, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_re, -1 / sqrt2 + 0.0j, atol=1e-15)
    npt.assert_allclose(e_im, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_im, 0.0 + 0.0j, atol=1e-15)

    # Magnetic dipole along Y
    e_re, b_re, e_im, b_im = get_coeffs("hertzian_h_dipole_y.sph", 1, 1)

    npt.assert_allclose(e_re, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_re, 1j / sqrt2, atol=1e-15)
    npt.assert_allclose(e_im, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_im, 0.0 + 0.0j, atol=1e-15)

    # Magnetic dipole along Z
    e_re, b_re, e_im, b_im = get_coeffs("hertzian_h_dipole_z.sph", 1, 0)

    npt.assert_allclose(e_re, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_re, 1.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(e_im, 0.0 + 0.0j, atol=1e-15)
    npt.assert_allclose(b_im, 0.0 + 0.0j, atol=1e-15)
