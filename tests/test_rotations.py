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

import numpy as np
import pytest

import sweaver

from utils import get_gaussian_beam, angular_distance_periodic


def test_rotate_euler():
    """Verifies that rotate_euler rotates the beam correctly by evaluating its peak."""
    gaussian_efield = get_gaussian_beam()

    base_th, base_ph, base_ps = gaussian_efield.find_peak(
        region_theta_rad=(0, np.radians(2.0), 10),
        region_phi_rad=(0, 2 * np.pi, 20),
    )
    assert base_th == pytest.approx(0.0, abs=1e-3)

    theta_off = np.radians(15.0)
    phi_off = np.radians(45.0)
    psi_off = np.radians(10.0)

    rotated_efield = gaussian_efield.rotate_euler(
        sweaver.EulerAngles(
            alpha_rad=phi_off,
            beta_rad=theta_off,
            gamma_rad=psi_off,
        ),
    )

    new_th, new_ph, new_ps = rotated_efield.find_peak(
        region_theta_rad=(np.radians(10.0), np.radians(20.0), 20),
        region_phi_rad=(np.radians(30.0), np.radians(60.0), 20),
    )

    assert new_th == pytest.approx(theta_off, abs=1e-3)
    assert angular_distance_periodic(new_ph, phi_off) == pytest.approx(0.0, abs=1e-3)


def test_rotate_grasp():
    """Verifies that rotate_grasp maps GRASP coordinates correctly to Euler angles."""
    gaussian_efield = get_gaussian_beam()

    theta_off = np.radians(15.0)
    phi_off = np.radians(45.0)
    psi_off = np.radians(10.0)

    # Inject the displacement using the GRASP rotation method
    rotated_efield = gaussian_efield.rotate_grasp(
        theta_rad=theta_off, phi_rad=phi_off, psi_rad=psi_off
    )

    # Evaluate the peak of the rotated field
    new_th, new_ph, new_ps = rotated_efield.find_peak(
        region_theta_rad=(np.radians(10.0), np.radians(20.0), 20),
        region_phi_rad=(np.radians(30.0), np.radians(60.0), 20),
    )

    # According to GRASP coordinate mapping:
    # 1. We first rotate around Z by phi -> This moves the "meridian" of our tilt.
    # 2. We tilt by theta -> The peak moves to `theta` colatitude and `phi` longitude.
    # Therefore, the peak location (th, ph) must perfectly match (theta_rad, phi_rad).
    assert new_th == pytest.approx(theta_off, abs=1e-3)
    assert new_ph == pytest.approx(phi_off, abs=1e-3)

    # The polarization twist `ps` should match the injected `psi_rad`.
    # Because linear polarization has a 180-degree (pi radians) physical symmetry,
    # we evaluate the difference modulo pi.
    diff_rad = new_ps - psi_off
    wrapped_diff = (diff_rad + np.pi / 2) % np.pi - (np.pi / 2)
    assert wrapped_diff == pytest.approx(0.0, abs=1e-3)
