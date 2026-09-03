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

import sweaver

import numpy as np

import pytest


def test_to_texture_shapes(synthetic_efield):
    """Verify that the output shape matches the requested equirectangular grid."""
    # Assuming 'efield' is a valid ElectricField object
    shape = (180, 360)
    texture = synthetic_efield.to_texture(shape=shape, mode="intensity")

    assert texture.shape == shape
    assert texture.dtype == np.float64


@pytest.mark.parametrize("mode_name", sweaver.MapMode._REGISTRY.keys())
def test_to_texture_modes(synthetic_efield, mode_name):
    """Ensure all supported modes return valid numerical data without NaNs."""

    shape = (50, 100)
    texture = synthetic_efield.to_texture(shape=shape, mode=mode_name)
    assert not np.isnan(texture).any(), f"NaN values found in mode {mode_name}"
    assert texture.shape == shape


def test_to_texture_intensity_positivity(synthetic_efield):
    """Intensity must always be a non-negative real number."""
    texture = synthetic_efield.to_texture(mode="intensity")
    assert np.all(texture >= 0)


def test_to_texture_phase_range(synthetic_efield):
    """Phase should stay within the standard [-pi, pi] branch cut."""
    texture = synthetic_efield.to_texture(mode="phase_theta")
    assert np.all(texture >= -np.pi)
    assert np.all(texture <= np.pi)


def test_to_texture_invalid_mode(synthetic_efield):
    """The API should fail gracefully if an unsupported mode is requested."""
    with pytest.raises(ValueError, match="Invalid mode"):
        synthetic_efield.to_texture(mode="not_a_real_mode")


def test_to_texture_periodicity(synthetic_efield):
    """In an equirectangular map, φ=0 and φ=2π should match (cylindrical wrap)."""
    shape = (100, 200)

    texture = synthetic_efield.to_texture(shape=shape, mode="re_theta")

    # Check if the start of the azimuthal range matches the end
    np.testing.assert_allclose(texture[:, 0], texture[:, -1], atol=1e-7)


def test_to_texture_energy_conservation(synthetic_efield):
    """
    The integral of the intensity texture over the sphere should match the power
    in the alm_stack (Parseval's Theorem).
    """
    n_theta, n_phi = 500, 1000
    texture = synthetic_efield.to_texture(shape=(n_theta, n_phi), mode="intensity")

    # Compute surface element sin(ϑ) dϑ dφ
    theta = np.linspace(0, np.pi, n_theta)
    d_theta = np.pi / n_theta
    d_phi = 2 * np.pi / n_phi
    sin_theta = np.sin(theta).reshape(-1, 1)

    integral = np.sum(texture * sin_theta) * d_theta * d_phi

    # Power in SH coefficients: sum(|alm|^2)
    coeffs_power = np.sum(np.abs(synthetic_efield.alm_stack) ** 2)

    # Allow for some discretization error based on grid resolution
    assert integral == pytest.approx(coeffs_power, rel=1e-2)
