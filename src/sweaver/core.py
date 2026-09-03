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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, TextIO

import numpy as np
import ducc0
import scipy.optimize
import scipy.constants
from scipy.special import roots_legendre

from .io import FrequencyBlock, read_sph_frequency_block
from .coord_sys import EulerAngles, CoordinateSystem


def _sph_to_cart(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian unit vectors.

    Args:
        theta: Colatitude angle (ϑ) in radians. May have arbitrary shape.
        phi: Azimuth angle (φ) in radians. Must have the same shape as theta.

    Returns:
        Array with shape ``(3, *theta.shape)`` containing Cartesian unit vectors.
    """
    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=0,
    )


def _theta_hat(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Return the local spherical e_ϑ basis vector.

    The result is the derivative direction associated with increasing ϑ
    at fixed φ. This works also for negative theta values, which is useful
    for TICRA-style cuts spanning negative and positive theta.
    """
    return np.stack(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ],
        axis=0,
    )


def _phi_hat(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Return the local spherical e_φ basis vector.

    The result is the derivative direction associated with increasing φ
    at fixed ϑ.
    """
    return np.stack(
        [
            -np.sin(phi),
            np.cos(phi),
            np.zeros_like(phi),
        ],
        axis=0,
    )


def _cart_to_sph(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian unit vectors to canonical spherical coordinates.

    Args:
        n: Array with shape ``(3, ...)``.

    Returns:
        Tuple ``(theta, phi)``, where ``theta`` is in ``[0, pi]`` and
        ``phi`` is in ``[0, 2*pi)``.
    """
    x, y, z = n

    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    return theta, phi


def _canonicalize_theta_phi(
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Canonicalize spherical coordinates for use with ducc0.

    SWEaver/TICRA cuts may use negative ϑ values. The direction

        (ϑ, φ), ϑ < 0

    is mapped to

        (-ϑ, φ + π)

    before calling ducc0, and the returned tangent components must then be
    multiplied by -1. The returned mask marks the samples where this happened.

    Args:
        theta: Input colatitude angles (ϑ).
        phi: Input azimuth angles (φ).

    Returns:
        ``theta_work, phi_work, neg_mask``.
    """
    theta_work = np.array(theta, copy=True)
    phi_work = np.array(phi, copy=True)

    neg_mask = theta_work < 0

    theta_work[neg_mask] = -theta_work[neg_mask]
    phi_work[neg_mask] = phi_work[neg_mask] + np.pi
    phi_work = np.mod(phi_work, 2 * np.pi)

    return theta_work, phi_work, neg_mask


class Polarization(Enum):
    THETA_PHI = "theta_phi"
    LUDWIG3_X = "ludwig3_x"
    LUDWIG3_Y = "ludwig3_y"


def _apply_polarization(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    phi_grid: np.ndarray,
    polarization: Polarization,
):
    """
    Applies the polarization projection matrix.

    Args:
        e_theta, e_phi (complex array): Field components.
        phi_grid (array): Azimuthal angles in radians (must match shape of e_theta).
        polarization (Polarization): The target polarization definition.

    Returns:
        (comp1, comp2): Transformed components.
    """
    if polarization == Polarization.THETA_PHI:
        return e_theta, e_phi

    sin_phi = np.sin(phi_grid)
    cos_phi = np.cos(phi_grid)

    if polarization == Polarization.LUDWIG3_X:
        # Source aligned with X-axis (Horizontal)
        # E_co = E_ϑ * cos(φ) - E_φ * sin(φ)
        # E_cx = E_ϑ * sin(φ) + E_φ * cos(φ)
        e_co = e_theta * cos_phi - e_phi * sin_phi
        e_cx = e_theta * sin_phi + e_phi * cos_phi
        return e_co, e_cx
    elif polarization == Polarization.LUDWIG3_Y:
        # Source aligned with Y-axis (Vertical)
        # E_co = E_ϑ * sin(φ) + E_φ * cos(φ)
        # E_cx = E_ϑ * cos(φ) - E_φ * sin(φ)
        e_co = e_theta * sin_phi + e_phi * cos_phi
        e_cx = e_theta * cos_phi - e_phi * sin_phi
        return e_co, e_cx
    else:
        raise NotImplementedError(f"Polarization {polarization} not supported")


MapCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]


class MapMode:
    """
    A collection of static mapping functions to transform complex electric field
    components into scalar values for visualization and analysis.

    This class serves as a namespace for standard transformations. Each method
    takes the complex :math:`E_\\theta` and :math:`E_\\phi` components (as
    NumPy arrays) and returns a single real-valued array of the same shape.
    """

    @staticmethod
    def intensity(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """
        Calculate the total power intensity of the field.

        Formula: :math:`I = |E_\\theta|^2 + |E_\\phi|^2`
        """
        return np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2

    @staticmethod
    def amplitude(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """
        Calculate the total magnitude of the electric field vector.

        Formula: :math:`A = \\sqrt{|E_\\theta|^2 + |E_\\phi|^2}`
        """
        return np.sqrt(np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2)

    @staticmethod
    def phase_theta(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Calculate the phase angle of the ϑ component in radians."""
        return np.angle(e_theta)

    @staticmethod
    def phase_phi(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Calculate the phase angle of the :math:`\\phi` component in radians."""
        return np.angle(e_phi)

    @staticmethod
    def re_theta(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Extract the real part of the ϑ component."""
        return np.real(e_theta)

    @staticmethod
    def im_theta(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Extract the imaginary part of the ϑ component."""
        return np.imag(e_theta)

    @staticmethod
    def re_phi(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Extract the real part of the φ component."""
        return np.real(e_phi)

    @staticmethod
    def im_phi(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Extract the imaginary part of the φ component."""
        return np.imag(e_phi)

    @staticmethod
    def db(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """
        Calculate the intensity in decibels (dB), normalized to the peak
        value within the provided field arrays.

        Formula: :math:`10 \\log_{10}(I / I_{\\text{max}})`
        """
        int_map = MapMode.intensity(e_theta, e_phi)
        return 10 * np.log10(int_map / (np.max(int_map) + 1e-20))

    # Internal registry for the 'to_texture' method
    _REGISTRY: dict[str, MapCallable] = {
        "intensity": intensity,
        "amplitude": amplitude,
        "phase_theta": phase_theta,
        "phase_phi": phase_phi,
        "re_theta": re_theta,
        "im_theta": im_theta,
        "re_phi": re_phi,
        "im_phi": im_phi,
        "db": db,
    }

    @classmethod
    def list_modes(cls) -> list[str]:
        """Return a list of all registered mapping mode names."""
        return list(cls._REGISTRY.keys())


class ElectricField:
    """
    An electric field represented using a set of spin-1 spherical harmonics.

    This class is the core of the library, providing the bridge between the raw
    spherical wave expansion coefficients from GRASP and a manipulable,
    physical representation of the beam. It stores the field as a set of
    spin-1 spherical harmonic coefficients (:math:`a_{\\ell m}`) and provides methods
    to transform, project, and analyze the beam.

    The coefficients are derived from the GRASP :math:`Q_{smn}` coefficients and
    normalized to represent the far-field electric field vector. This class
    serves as the foundation for all subsequent operations, including conversion
    to Stokes parameters, rotation, and visualization.

    Attributes:
        frequency_ghz (float): The frequency of the monochromatic field in GHz.
        lmax (int): The maximum multipole order :math:`\\ell` of the expansion.
        mmax (int): The maximum azimuthal order :math:`m` of the expansion.
        alm_stack (np.ndarray): A NumPy array of shape `(4, nalm)` containing
            the complex spherical harmonic coefficients. These represent the
            real and imaginary parts of the electric and magnetic potentials.
    """

    def __init__(
        self,
        frequency_ghz: float,
        lmax: int,
        mmax: int,
        alm_stack: np.ndarray,
    ) -> None:
        self.frequency_ghz = frequency_ghz
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.alm_stack = np.asarray(alm_stack, dtype=np.complex128)

        if self.lmax < self.mmax:
            raise ValueError(
                f"lmax must be >= mmax; got lmax={self.lmax}, mmax={self.mmax}"
            )

        expected_shape = (4, ElectricField._num_of_alms(self.lmax, self.mmax))

        if self.alm_stack.shape != expected_shape:
            raise ValueError(
                "alm_stack has shape inconsistent with lmax/mmax; "
                f"got {self.alm_stack.shape}, expected {expected_shape} "
                f"for lmax={self.lmax}, mmax={self.mmax}"
            )

    @classmethod
    def from_frequency_block(cls, freq_block: FrequencyBlock):
        """
        Create an ElectricField instance from a FrequencyBlock.

        This class method converts the raw :math:`Q_{smn}` coefficients from a
        :class:`.FrequencyBlock` into the spin-1 spherical harmonic
        coefficients (:math:`a_{\\ell m}`) that represent the electric field.
        The coefficients are normalized to ensure consistency with standard
        spherical harmonic conventions.

        Args:
            freq_block (FrequencyBlock): An instance of :class:`.FrequencyBlock` containing the
                GRASP spherical wave expansion coefficients for a single frequency.

        Returns:
            ElectricField: A new `ElectricField` instance initialized with the
                converted spherical harmonic coefficients.
        """
        nalm = ElectricField._num_of_alms(
            freq_block.header.lmax, freq_block.header.mmax
        )
        alm_stack = np.zeros((4, nalm), dtype=np.complex128)

        ElectricField._build_alms_from_q(
            freq_block=freq_block,
            alm_stack=alm_stack,
        )

        return cls(
            frequency_ghz=freq_block.header.frequency_ghz,
            lmax=freq_block.header.lmax,
            mmax=freq_block.header.mmax,
            alm_stack=alm_stack,
        )

    @staticmethod
    def _num_of_alms(lmax: int, mmax: int) -> int:
        return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)

    @staticmethod
    def _get_idx(ell: int, m: int, lmax: int) -> int:
        """Return the index of an a_ℓm coefficient in a Healpix/ducc0 array."""
        return m * (2 * lmax + 1 - m) // 2 + ell

    @staticmethod
    def _coor_sys_to_rotation_matrix(
        coor_sys: CoordinateSystem | EulerAngles | np.ndarray,
    ) -> np.ndarray:
        """
        Convert a coordinate-system-like object to a child-to-base rotation matrix.
        """
        if isinstance(coor_sys, CoordinateSystem):
            return coor_sys.as_child_to_base_matrix()

        if isinstance(coor_sys, EulerAngles):
            return coor_sys.as_child_to_base_matrix()

        return np.asarray(coor_sys, dtype=float)

    @staticmethod
    def analyze_gl_grid_to_alm(
        grid_E_theta: np.ndarray,
        grid_E_phi: np.ndarray,
        lmax: int,
        mmax: int,
        spin: int = 1,
        nthreads: int = 1,
    ) -> np.ndarray:
        """
        Analyzes a complex electric field evaluated on a Gauss-Legendre grid
        back into spherical harmonic coefficients (a_lm).

        Args:
            grid_E_theta (np.ndarray): Complex E-field theta component, shape (nlat, nlon).
            grid_E_phi (np.ndarray): Complex E-field phi component, shape (nlat, nlon).
            lmax (int): Maximum multipole.
            mmax (int): Maximum azimuthal mode.
            spin (int): Spin-weight of the field (1 for electric field vectors).
            nthreads (int): Number of threads for ducc0 parallelization.

        Returns:
            np.ndarray: Complex array of shape (4, nalm) containing:
                        [0, :] -> E-mode of Real(E)
                        [1, :] -> B-mode of Real(E)
                        [2, :] -> E-mode of Imag(E)
                        [3, :] -> B-mode of Imag(E)
        """

        map_real = np.ascontiguousarray([np.real(grid_E_theta), np.real(grid_E_phi)])
        map_imag = np.ascontiguousarray([np.imag(grid_E_theta), np.imag(grid_E_phi)])

        alm_real = ducc0.sht.analysis_2d(
            map=map_real,
            spin=spin,
            geometry="GL",
            lmax=lmax,
            mmax=mmax,
            nthreads=nthreads,
        )

        alm_imag = ducc0.sht.analysis_2d(
            map=map_imag,
            spin=spin,
            geometry="GL",
            lmax=lmax,
            mmax=mmax,
            nthreads=nthreads,
        )

        nalm = alm_real.shape[1]
        alm_stack = np.empty((4, nalm), dtype=np.complex128)

        alm_stack[0, :] = alm_real[0, :]  # E-mode of the Real part
        alm_stack[1, :] = alm_real[1, :]  # B-mode of the Real part
        alm_stack[2, :] = alm_imag[0, :]  # E-mode of the Imaginary part
        alm_stack[3, :] = alm_imag[1, :]  # B-mode of the Imaginary part

        return alm_stack

    def get_alms(
        self, ell: int, m: int
    ) -> tuple[np.complex128, np.complex128, np.complex128, np.complex128]:
        idx = ElectricField._get_idx(ell, m, lmax=self.lmax)
        return tuple(self.alm_stack[:, idx])

    @staticmethod
    def _build_alms_from_q(
        freq_block: FrequencyBlock,
        alm_stack: np.ndarray,
    ) -> None:
        # This normalizes the beam to 4π
        scale_factor = np.sqrt(4 * np.pi)

        lmax = freq_block.header.lmax
        mmax = freq_block.header.mmax

        # As the Electric field is a spin-1 field, we skip ℓ=0 (the monopole)
        for ell in range(1, lmax + 1):
            for m in range(mmax + 1):
                idx = ElectricField._get_idx(ell, m, lmax=lmax)

                q1_mpos = scale_factor * freq_block.get_q(s=1, m=m, n=ell)
                q2_mpos = scale_factor * freq_block.get_q(s=2, m=m, n=ell)
                q1_mneg = scale_factor * freq_block.get_q(s=1, m=-m, n=ell)
                q2_mneg = scale_factor * freq_block.get_q(s=2, m=-m, n=ell)

                j_ell = (-1j) ** ell  # (−j)ⁿ
                j_ell_1 = j_ell * (-1j)  # (−j)ⁿ⁺¹

                phase_sym = (-1) ** (ell + m)

                alm_stack[0, idx] = (
                    0.5 * j_ell * (q2_mpos + phase_sym * np.conj(q2_mneg))
                )
                alm_stack[1, idx] = (
                    -0.5 * j_ell_1 * (q1_mpos + (-1) * phase_sym * np.conj(q1_mneg))
                )

                alm_stack[2, idx] = (
                    0.5 * j_ell_1 * (q2_mpos - phase_sym * np.conj(q2_mneg))
                )
                alm_stack[3, idx] = (
                    0.5 * j_ell * (q1_mpos - (-1) * phase_sym * np.conj(q1_mneg))
                )

    def total_power(self) -> float:
        """
        Compute the total integrated power of the electric field over the full sphere
        using the spherical harmonic coefficients.

        Returns ∫|E|² dΩ.
        """
        # Calculate the absolute square of every coefficient in the stack
        # This covers Real E, Real B, Imag E, and Imag B components.
        power_stack = np.abs(self.alm_stack) ** 2

        # Create a weights array for the 1D ducc0 layout, as we need to treat m=0
        # differently (see below)
        nalm = power_stack.shape[1]
        weights = np.full(nalm, 2.0, dtype=np.float64)

        # The m=0 modes do not have a negative counterpart, so their weight is exactly 1.0.
        # In the ducc0 memory layout, the m=0 chunk is always the first (lmax + 1) elements.
        weights[0 : self.lmax + 1] = 1.0

        integral_E_squared = np.sum(power_stack * weights)

        return float(integral_E_squared)

    def _pad_alm_stack(self, target_lmax: int, target_mmax: int) -> np.ndarray:
        """
        Safely pads the current a_lm array with zeros up to a new, larger target
        lmax and mmax by copying contiguous m-chunks according to the ducc0 memory layout.
        """
        if target_lmax < self.lmax or target_mmax < self.mmax:
            raise ValueError(
                "Target dimensions must be greater than or equal to current dimensions."
            )

        if target_mmax > target_lmax:
            raise ValueError(
                f"target_mmax must be ≤ target_lmax; "
                f"got target_mmax={target_mmax}, target_lmax={target_lmax}"
            )

        # Correct calculation of total size for ducc0/HEALPix layout
        # This is the sum of (target_lmax - m + 1) from m=0 to target_mmax
        nalm_new = (target_mmax + 1) * (2 * target_lmax + 2 - target_mmax) // 2

        new_stack = np.zeros((4, nalm_new), dtype=np.complex128)

        for m in range(self.mmax + 1):
            # The number of multipoles for this m in the OLD array
            l_count = self.lmax - m + 1

            # The exact mathematical start index for chunk 'm' in ducc0
            idx_old_start = m * (2 * self.lmax + 3 - m) // 2
            idx_old_end = idx_old_start + l_count

            # The exact mathematical start index for chunk 'm' in the PADDED array
            idx_new_start = m * (2 * target_lmax + 3 - m) // 2
            idx_new_end = idx_new_start + l_count

            # Fast numpy contiguous copy
            new_stack[:, idx_new_start:idx_new_end] = self.alm_stack[
                :, idx_old_start:idx_old_end
            ]

        return new_stack

    def with_lmax_mmax(
        self,
        lmax: int | None = None,
        mmax: int | None = None,
    ) -> "ElectricField":
        """
        Return an equivalent field represented with larger harmonic limits.

        This method returns a new spherical-electric-field object whose harmonic
        coefficients are defined up to the requested ``lmax`` and ``mmax``. Existing
        coefficients are copied into the appropriate ducc0/HEALPix-layout positions,
        and all newly introduced coefficients are set to zero.

        The represented physical field is unchanged. Only the size of the harmonic
        coefficient storage changes.

        This is useful when comparing or combining fields whose coefficients were
        produced with different ``mmax`` values. In particular, an axially symmetric
        or otherwise low-``mmax`` field may acquire non-zero coefficients at higher
        ``m`` after an arbitrary rotation. Therefore, rotated and unrotated versions
        of the same field may need to be promoted to a common harmonic space before
        their raw coefficient arrays can be compared.

        Args:
            lmax:
                Target maximum multipole. If ``None``, the current ``self.lmax`` is
                used.

            mmax:
                Target maximum azimuthal order. If ``None``, the current
                ``self.mmax`` is used.

        Returns:
            A new field object with the same physical content but with harmonic
            storage large enough for the requested ``lmax`` and ``mmax``.

        Raises:
            ValueError:
                If ``lmax`` or ``mmax`` is smaller than the current value, or if
                ``mmax > lmax``.
        """
        target_lmax: int = self.lmax if lmax is None else int(lmax)
        target_mmax: int = self.mmax if mmax is None else int(mmax)

        if target_lmax < self.lmax:
            raise ValueError(
                f"target lmax must be ≥ current lmax; got {target_lmax} < {self.lmax}"
            )

        if target_mmax < self.mmax:
            raise ValueError(
                f"target mmax must be ≥ current mmax; got {target_mmax} < {self.mmax}"
            )

        if target_mmax > target_lmax:
            raise ValueError(
                f"target mmax must be ≤ target lmax; "
                f"got mmax={target_mmax}, lmax={target_lmax}"
            )

        if target_lmax == self.lmax and target_mmax == self.mmax:
            return ElectricField(
                frequency_ghz=self.frequency_ghz,
                lmax=self.lmax,
                mmax=self.mmax,
                alm_stack=self.alm_stack.copy(),
            )

        new_alm_stack = self._pad_alm_stack(
            target_lmax=target_lmax,
            target_mmax=target_mmax,
        )

        return ElectricField(
            frequency_ghz=self.frequency_ghz,
            lmax=target_lmax,
            mmax=target_mmax,
            alm_stack=new_alm_stack,
        )

    def compatible_with(
        self, other: "ElectricField"
    ) -> tuple[
        "ElectricField",
        "ElectricField",
    ]:
        """
        Return two equivalent fields promoted to a common harmonic representation.

        The common representation uses

            lmax = max(self.lmax, other.lmax)
            mmax = max(self.mmax, other.mmax)

        Existing coefficients are preserved and missing coefficients are filled with
        zero. This is useful before comparing raw ``alm_stack`` arrays or performing
        operations that require identical storage layouts.
        """
        target_lmax = max(self.lmax, other.lmax)
        target_mmax = max(self.mmax, other.mmax)

        return (
            self.with_lmax_mmax(lmax=target_lmax, mmax=target_mmax),
            other.with_lmax_mmax(lmax=target_lmax, mmax=target_mmax),
        )

    def _check_compatible_frequency(self, other: "ElectricField") -> None:
        if not np.isclose(self.frequency_ghz, other.frequency_ghz, atol=1e-6):
            raise ValueError(
                "Cannot combine fields with different physical frequencies: "
                f"{self.frequency_ghz} GHz != {other.frequency_ghz} GHz"
            )

    def __add__(self, other: "ElectricField") -> "ElectricField":
        """Allows algebraic addition of two ElectricFields: field3 = field1 + field2"""
        self._check_compatible_frequency(other)

        self_cmp, other_cmp = self.compatible_with(other)

        return ElectricField(
            frequency_ghz=self_cmp.frequency_ghz,
            lmax=self_cmp.lmax,
            mmax=self_cmp.mmax,
            alm_stack=self_cmp.alm_stack + other_cmp.alm_stack,
        )

    def __sub__(self, other: "ElectricField") -> "ElectricField":
        """Allows algebraic subtraction of two ElectricFields: field3 = field1 - field2"""
        self._check_compatible_frequency(other)

        self_cmp, other_cmp = self.compatible_with(other)

        return ElectricField(
            frequency_ghz=self_cmp.frequency_ghz,
            lmax=self_cmp.lmax,
            mmax=self_cmp.mmax,
            alm_stack=self_cmp.alm_stack - other_cmp.alm_stack,
        )

    def project_to_gl(
        self,
        n_theta: int | None = None,
        n_phi: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project the spherical harmonic expansion over a Gauss-Legendre grid

        Return a tuple `(E_theta, E_phi)` containing the components of the far-field
        electric vector decomposed along the ϑ/φ axes. The two arrays `E_theta` and `E_phi`
        have shape `(N, M)`, where `N` is the number of values along the ϑ direction
        and `M` the number of values along the φ direction.
        """

        # Define the Gauss-Legendre grid
        if not n_theta:
            n_theta = self.lmax + 10

        if not n_phi:
            n_phi = 2 * n_theta

        # Real part of the phasor
        map_vec_re = ducc0.sht.synthesis_2d(
            alm=self.alm_stack[0:2],
            spin=1,
            ntheta=n_theta,
            nphi=n_phi,
            geometry="GL",
            lmax=self.lmax,
            mmax=self.mmax,
        )

        # Imaginary part of the phasor
        map_vec_im = ducc0.sht.synthesis_2d(
            alm=self.alm_stack[2:4],
            spin=1,
            ntheta=n_theta,
            nphi=n_phi,
            geometry="GL",
            lmax=self.lmax,
            mmax=self.mmax,
        )

        efield_theta = map_vec_re[0] + 1j * map_vec_im[0]
        efield_phi = map_vec_re[1] + 1j * map_vec_im[1]

        return (efield_theta, efield_phi)

    def rotate_euler(self, angles: EulerAngles) -> "ElectricField":
        """
        Return a rotated copy of the beam. The rotation is expressed
        using standard Euler angles (Z-Y-Z convention).

        Args:
            angles: An instance of the class :class:`.EulerAngles`

        Returns:
            ElectricField: A new, rotated field object.
        """

        # As Ducc use the *extrinsic* definition of Euler angles but `EulerAngles`
        # follows Ticra’s *intrinsic* convention, we must reverse the list of
        # angles: from φ↔α, θ↔β, ψ↔γ to ψ↔α, θ↔β, φ↔γ
        alm_rotated = ducc0.sht.rotate_alm(
            alm=self.alm_stack,
            lmax=self.lmax,
            mmax_in=self.mmax,
            mmax_out=self.lmax,  # As the rotation might disrupt symmetry, we do *not* use self.mmax here!
            phi=angles.alpha_rad,
            theta=angles.beta_rad,
            psi=angles.gamma_rad,
        )

        return ElectricField(
            frequency_ghz=self.frequency_ghz,
            lmax=self.lmax,
            mmax=self.lmax,  # Sic!
            alm_stack=alm_rotated,
        )

    def rotate_grasp(
        self, theta_rad: float, phi_rad: float, psi_rad: float
    ) -> "ElectricField":
        """
        Rotate the beam using the specific coordinate system parameters
        defined in a TICRA GRASP project.

        This method safely maps GRASP's (ϑ, φ, ψ) parameters to the
        standard Z-Y-Z active Euler angles used by SWEaver.

        Because the IAU polarization convention evaluates the twist looking at
        the *sky* (−r vector), and TICRA evaluates it looking *outward* (+r vector)
        using a clockwise definition, the two minus signs cancel. The TICRA
        parameters map 1:1 to the active Wigner rotations:
        1. α (inner twist)    = φ
        2. β (tilt)           = ϑ
        3. γ (polarization)   = ψ

        Args:
            theta_rad (float): The GRASP 'theta' parameter.
            phi_rad (float):   The GRASP 'phi' parameter.
            psi_rad (float):   The GRASP 'psi' parameter.
        """
        # The parameter `alpha_rad` has no minus sign because of the IAU/CMB mismatch
        angles = EulerAngles(
            alpha_rad=phi_rad,
            beta_rad=theta_rad,
            gamma_rad=psi_rad,
        )

        return self.rotate_euler(angles)

    def translate_phase_center(
        self,
        dx_m: float,
        dy_m: float,
        dz_m: float,
        lmax_out: int | None = None,
        mmax_out: int | None = None,
    ) -> "ElectricField":
        """
        Shift the phase center of the far-field beam by a vector (dx, dy, dz) in meters

        This routine uses exact Gauss-Legendre quadrature to apply the translation
        phase factor, so it does not introduce integration errors in the procedure.

        To prevent spatial aliasing, the field is upsampled to a new harmonic bandwidth
        l_new = l_old + k|d| before the phase factor is applied.
        """

        #  Calculate the physical shift bandwidth
        d_mag = np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
        wavelength_m = scipy.constants.speed_of_light / (self.frequency_ghz * 1e9)
        k0 = 2 * np.pi / wavelength_m

        # Add an asymptotic buffer to capture the exponential tail of the Bessel functions,
        # pushing the truncation error down to the 64-bit machine precision floor.
        # This mirrors the truncation rules used by TICRA Tools
        if d_mag > 0.0:
            kd = k0 * d_mag  # The baseline physical bandwidth
            padding = int(np.ceil(3.6 * np.cbrt(kd))) + 15
            l_shift = int(np.ceil(kd)) + padding
        else:
            l_shift = 0

        # Determine new truncation limits (use user inputs if provided, else use physical rules)
        l_new = lmax_out if lmax_out is not None else self.lmax + l_shift

        # If the shift is purely along Z (dx=0, dy=0), m modes do not mix.
        # Otherwise, mmax must grow to capture the broken symmetry.
        # The point is that we must sample the field over a grid dense enough
        # to capture l_new to prevent aliasing
        if mmax_out is not None:
            m_new = mmax_out
        elif dx_m == 0.0 and dy_m == 0.0:
            m_new = self.mmax
        else:
            m_new = l_new

        nlat = l_new + 1
        nlon = 2 * l_new + 1

        grid_E_theta, grid_E_phi = self.project_to_gl(n_theta=nlat, n_phi=nlon)

        # Calculate spatial phase shift. As `roots_legendre` returns (nodes, weights),
        # we discard `weights`. Also, `nodes` are sorted from −1 to 1 (South to North),
        # but Ducc evaluates grids from North to South, so we must reverse the array
        nodes, _ = roots_legendre(nlat)
        colat = np.arccos(nodes[::-1])
        lon = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        phi, theta = np.meshgrid(lon, colat)

        rx = np.sin(theta) * np.cos(phi)
        ry = np.sin(theta) * np.sin(phi)
        rz = np.cos(theta)

        phase = k0 * (rx * dx_m + ry * dy_m + rz * dz_m)
        shift_factor = np.exp(1j * phase)

        shifted_E_theta = grid_E_theta * shift_factor
        shifted_E_phi = grid_E_phi * shift_factor

        new_alm = ElectricField.analyze_gl_grid_to_alm(
            shifted_E_theta,
            shifted_E_phi,
            lmax=l_new,
            mmax=m_new,
            spin=1,
        )

        return ElectricField(self.frequency_ghz, l_new, m_new, new_alm)

    def expressed_in_coordinate_system(
        self,
        source_csy_in_target: CoordinateSystem,
    ) -> "ElectricField":
        """
        Return this field transformed from its local/source coordinate system into
        a target coordinate system.

        The argument describes the source coordinate system expressed relative to
        the target coordinate system. In other words, if this field's coefficients
        are currently expressed in frame S, and the desired output coefficients
        should be expressed in frame T, then pass

            source_csy_in_target = source_csy.relative_to(target_csy)

        The transformation is active: the local field is rotated and translated
        into the target frame so that the returned SWE coefficients can be summed
        with other fields expressed in the same target frame.
        """
        return self.rotate_euler(source_csy_in_target.angles).translate_phase_center(
            *source_csy_in_target.origin_m
        )

    def evaluate_at_locs(
        self,
        theta_rad: np.ndarray,
        phi_rad: np.ndarray,
        polarization: Polarization,
        epsilon: float = 1e-8,
        use_ticra_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the electric field at arbitrary spherical-coordinate locations.

        This is the most general single-frame evaluator. Unlike
        :meth:`evaluate_theta_phi_grid`, the input directions do not need to lie on
        a regular tensor-product grid. The arrays ``theta_rad`` and ``phi_rad`` may
        have any shape, provided they have the same shape.

        The supplied coordinates define both:

        1. the physical directions where the field is sampled;
        2. the local spherical basis used for the returned field components.

        Therefore, when ``polarization`` is ``Polarization.LUDWIG3_X`` or
        ``Polarization.LUDWIG3_Y``, the Ludwig-3 projection is computed using the
        same azimuthal angle ``phi_rad`` passed to this method.

        This method is appropriate when the sampling directions and the requested
        output polarization basis belong to the same coordinate system. It is not,
        by itself, sufficient for evaluating a cut defined in a different coordinate
        system from the one in which the SWE coefficients are expressed. For that
        case, use :meth:`evaluate_in_frame`.

        Negative values of ``theta_rad`` are supported using the same convention as
        :meth:`evaluate_theta_phi_grid`: a sample at ``(theta, phi)`` with
        ``theta < 0`` is evaluated as ``(-theta, phi + pi)`` and the tangent-vector
        components are sign-flipped afterwards.

        Args:
            theta_rad:
                Colatitude angles in radians. May be any NumPy-broadcasted array,
                but must have exactly the same shape as ``phi_rad`` after
                conversion to arrays.

            phi_rad:
                Azimuthal angles in radians. Must have the same shape as
                ``theta_rad``.

            polarization:
                Output polarization basis. If ``Polarization.THETA_PHI``, the
                returned components are ``(E_theta, E_phi)`` in the local spherical
                basis associated with ``theta_rad, phi_rad``. If Ludwig-3 is
                requested, the returned components are ``(E_co, E_cx)``.

            epsilon:
                Desired accuracy for ``ducc0.sht.synthesis_general``.

            use_ticra_phase:
                If ``True``, complex-conjugate the returned components to match the
                TICRA GRASP convention used in ``.cut`` and ``.grd`` files.

        Returns:
            Tuple ``(comp1, comp2)`` of complex arrays with the same shape as the
            input angles.
        """
        theta = np.asarray(theta_rad)
        phi = np.asarray(phi_rad)

        if theta.shape != phi.shape:
            raise ValueError(
                "theta_rad and phi_rad must have the same shape; "
                f"got {theta.shape} and {phi.shape}"
            )

        theta_work, phi_work, neg_mask = _canonicalize_theta_phi(theta, phi)

        loc = np.stack((theta_work.ravel(), phi_work.ravel()), axis=-1)

        map_vec_re = ducc0.sht.synthesis_general(
            alm=self.alm_stack[0:2],
            spin=1,
            lmax=self.lmax,
            mmax=self.mmax,
            loc=loc,
            epsilon=epsilon,
        )

        map_vec_im = ducc0.sht.synthesis_general(
            alm=self.alm_stack[2:4],
            spin=1,
            lmax=self.lmax,
            mmax=self.mmax,
            loc=loc,
            epsilon=epsilon,
        )

        e_theta = (map_vec_re[0] + 1j * map_vec_im[0]).reshape(theta.shape)
        e_phi = (map_vec_re[1] + 1j * map_vec_im[1]).reshape(theta.shape)

        # Convert components back to the tangent basis associated with the
        # original possibly-negative theta parameterization.
        e_theta[neg_mask] = -e_theta[neg_mask]
        e_phi[neg_mask] = -e_phi[neg_mask]

        result = _apply_polarization(
            e_theta=e_theta,
            e_phi=e_phi,
            phi_grid=phi,
            polarization=polarization,
        )

        if use_ticra_phase:
            return np.conj(result[0]), np.conj(result[1])

        return result

    def evaluate_theta_phi_grid(
        self,
        theta_start_rad: float,
        theta_end_rad: float,
        ntheta: int,
        phi_start_rad: float,
        phi_end_rad: float,
        nphi: int,
        polarization: Polarization,
        epsilon: float = 1e-8,
        use_ticra_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the electric field on an arbitrary 2D grid of spherical coordinates.

        This method projects the spherical harmonic coefficients onto a user-defined
        grid in real space, allowing for flexible sampling of the beam. The output
        components depend on the chosen polarization basis.

        Args:
            theta_start_rad (float): Starting colatitude ϑ (polar angle) in radians (-π to π).
            theta_end_rad (float): Ending colatitude ϑ (polar angle) in radians (-π to π).
            ntheta (int): Number of samples along the colatitude (ϑ) direction.
            phi_start_rad (float): Starting longitude φ (azimuthal angle) in radians (0 to 2π).
            phi_end_rad (float): Ending longitude φ (azimuthal angle) in radians (0 to 2π).
            nphi (int): Number of samples along the longitude (φ) direction.
            polarization (Polarization): The polarization basis to use for the output components
                (see :class:`.Polarization`).
            epsilon (float, optional): Desired accuracy for the spherical harmonic transform, by default 1e-8.
            use_ticra_phase (bool, optional): If ``True``, the complex conjugate of the field components is returned
                to match TICRA GRASP convention for ``.cut`` and ``.grd`` files.
                By default, ``False``.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two complex NumPy arrays, (Comp1, Comp2).
                Each array has shape ``(ntheta, nphi)`` and represents the
                field components in the specified polarization basis.
        """
        theta = np.linspace(theta_start_rad, theta_end_rad, num=ntheta)
        phi = np.linspace(phi_start_rad, phi_end_rad, num=nphi)

        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        return self.evaluate_at_locs(
            theta_rad=theta_grid,
            phi_rad=phi_grid,
            polarization=polarization,
            epsilon=epsilon,
            use_ticra_phase=use_ticra_phase,
        )

    def evaluate_cut(
        self,
        phi_angle_rad: float,
        theta_start_rad: float,
        theta_end_rad: float,
        ntheta: int,
        polarization: Polarization,
        epsilon=1e-8,
        use_ticra_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a 1D cut of the electric field at a constant azimuthal angle.

        This method evaluates the electric field along a specific meridian (constant phi)
        over a range of colatitudes. It's useful for analyzing beam patterns in a single plane.

        Args:
            phi_angle_rad (float): The constant azimuthal angle (longitude) in radians for the cut.
            theta_start_rad (float): Starting colatitude (polar angle) in radians (0 to pi).
            theta_end_rad (float): Ending colatitude (polar angle) in radians (0 to pi).
            ntheta (int): Number of samples along the colatitude (theta) direction for the cut.
            polarization (Polarization): The polarization basis to use for the output components
                (see :class:`.Polarization`).
            epsilon (float, optional): Desired accuracy for the spherical harmonic transform, by default 1e-8.
            use_ticra_phase (bool, optional): If ``True``, the complex conjugate of the field components is returned
                to match TICRA GRASP convention for ``.cut`` and ``.grd`` files.
                By default ``False``.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two complex NumPy arrays, (Comp1, Comp2).
                Each array has shape ``(ntheta,)`` and represents the
                field components along the 1D cut in the specified polarization basis.
        """

        e1, e2 = self.evaluate_theta_phi_grid(
            theta_start_rad=theta_start_rad,
            theta_end_rad=theta_end_rad,
            ntheta=ntheta,
            phi_start_rad=phi_angle_rad,
            phi_end_rad=phi_angle_rad,
            nphi=1,
            polarization=polarization,
            epsilon=epsilon,
            use_ticra_phase=use_ticra_phase,
        )

        return e1.flatten(), e2.flatten()

    def evaluate_in_frame(
        self,
        theta_rad: np.ndarray,
        phi_rad: np.ndarray,
        rotation_matrix_child_to_base: np.ndarray,
        polarization: Polarization,
        epsilon: float = 1e-8,
        use_ticra_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the electric field along directions defined in a rotated frame.

        This method evaluates the SWE coefficients, which are assumed to be
        expressed in the base frame, at directions whose spherical coordinates are
        given in a child/local coordinate system. It then projects the resulting
        vector field onto the local spherical basis of the child frame before
        applying the requested polarization convention.

        This is the appropriate operation for comparing against TICRA cuts of the
        form

            spherical_cut(
                coor_sys : ref(some_rotated_coordinate_system),
                ...
            )

        when the underlying field coefficients are still those of the parent/base
        coordinate system.

        The operation performed is:

        1. Interpret ``theta_rad, phi_rad`` as spherical coordinates in the child
           frame.
        2. Convert each local direction to a Cartesian unit vector in the child
           frame.
        3. Rotate those unit vectors into the base frame using
           ``rotation_matrix_child_to_base``.
        4. Convert the base-frame directions to spherical coordinates.
        5. Evaluate the SWE coefficients in the base-frame spherical basis.
        6. Reconstruct the vector electric field in base-frame Cartesian
           components.
        7. Rotate/project the field components onto the child-frame local spherical
           basis.
        8. Apply the requested polarization convention using the child-frame
           azimuth ``phi_rad``.

        This differs from :meth:`evaluate_at_locs`, where the same coordinate system
        is used both for sampling the field and for defining the output polarization
        basis. Here, the field is sampled in the base frame but reported in the
        child-frame basis.

        Args:
            theta_rad:
                Colatitude angles in radians in the child coordinate system. May
                have arbitrary shape, but must have the same shape as ``phi_rad``.
                Negative theta values are supported, as in TICRA-style cuts.

            phi_rad:
                Azimuthal angles in radians in the child coordinate system. Must
                have the same shape as ``theta_rad``.

            rotation_matrix_child_to_base:
                Real ``3x3`` rotation matrix mapping Cartesian vector components
                from the child frame to the base frame. In other words, if ``v_c``
                is a vector expressed in the child frame, then

                    v_b = rotation_matrix_child_to_base @ v_c

                is the same physical vector expressed in the base frame.

            polarization:
                Output polarization basis. If ``Polarization.THETA_PHI``, the
                returned components are ``(E_theta_child, E_phi_child)`` in the
                local spherical basis of the child frame. If Ludwig-3 is requested,
                the returned components are ``(E_co, E_cx)`` computed using the
                child-frame azimuth ``phi_rad``.

            epsilon:
                Desired accuracy for ``ducc0.sht.synthesis_general``.

            use_ticra_phase:
                If ``True``, complex-conjugate the returned components to match the
                TICRA GRASP convention used in ``.cut`` and ``.grd`` files.

        Returns:
            Tuple ``(comp1, comp2)`` of complex arrays with the same shape as
            ``theta_rad`` and ``phi_rad``.

        Raises:
            ValueError:
                If the input angle arrays do not have the same shape, or if
                ``rotation_matrix_child_to_base`` is not a ``3x3`` matrix.
        """
        theta_child = np.asarray(theta_rad)
        phi_child = np.asarray(phi_rad)

        if theta_child.shape != phi_child.shape:
            raise ValueError(
                "theta_rad and phi_rad must have the same shape; "
                f"got {theta_child.shape} and {phi_child.shape}"
            )

        R = np.asarray(rotation_matrix_child_to_base, dtype=float)

        if R.shape != (3, 3):
            raise ValueError(
                f"rotation_matrix_child_to_base must have shape (3, 3); got {R.shape}"
            )

        # Directions and local spherical basis vectors in the child frame.
        #
        # We intentionally use the original theta_child values here, including
        # possible negative values, because the basis associated with a TICRA-style
        # negative-theta cut is the derivative basis of that parameterization.
        n_child = _sph_to_cart(theta_child, phi_child)
        thhat_child = _theta_hat(theta_child, phi_child)
        phhat_child = _phi_hat(theta_child, phi_child)

        # Rotate directions and basis vectors from child frame to base frame.
        #
        # Shapes:
        #   R             : (3, 3)
        #   n_child       : (3, ...)
        #   n_base        : (3, ...)
        n_base = np.einsum("ij,j...->i...", R, n_child)
        thhat_child_in_base = np.einsum("ij,j...->i...", R, thhat_child)
        phhat_child_in_base = np.einsum("ij,j...->i...", R, phhat_child)

        # Convert the base-frame directions to canonical spherical coordinates.
        theta_base, phi_base = _cart_to_sph(n_base)

        # Evaluate in the base-frame spherical basis. Do not apply Ludwig-3 here:
        # the Ludwig-3 projection must be done in the child frame, after the vector
        # components have been projected onto the child-frame tangent basis.
        e_theta_base, e_phi_base = self.evaluate_at_locs(
            theta_rad=theta_base,
            phi_rad=phi_base,
            polarization=Polarization.THETA_PHI,
            epsilon=epsilon,
            use_ticra_phase=False,
        )

        # Base-frame spherical basis vectors at the sampled directions.
        thhat_base = _theta_hat(theta_base, phi_base)
        phhat_base = _phi_hat(theta_base, phi_base)

        # Reconstruct the electric field as a Cartesian vector in the base frame.
        e_cart_base = (
            e_theta_base[None, ...] * thhat_base + e_phi_base[None, ...] * phhat_base
        )

        # Project onto the child-frame local spherical basis, expressed in the base
        # frame. This is the crucial step absent from a simple rotated-coordinate
        # call to evaluate_at_locs().
        e_theta_child = np.sum(e_cart_base * thhat_child_in_base, axis=0)
        e_phi_child = np.sum(e_cart_base * phhat_child_in_base, axis=0)

        result = _apply_polarization(
            e_theta=e_theta_child,
            e_phi=e_phi_child,
            phi_grid=phi_child,
            polarization=polarization,
        )

        if use_ticra_phase:
            return np.conj(result[0]), np.conj(result[1])

        return result

    def evaluate_theta_phi_grid_in_frame(
        self,
        theta_start_rad: float,
        theta_end_rad: float,
        ntheta: int,
        phi_start_rad: float,
        phi_end_rad: float,
        nphi: int,
        frame: EulerAngles | np.ndarray,
        polarization: Polarization,
        epsilon: float = 1e-8,
        use_ticra_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the electric field on a regular angular grid defined in a rotated frame.

        This method is the frame-aware counterpart of
        :meth:`evaluate_theta_phi_grid`. The spherical coordinates ``theta`` and
        ``phi`` are interpreted as coordinates in a child/local coordinate system,
        while the SWE coefficients stored in this object are assumed to be expressed
        in the base coordinate system.

        The method evaluates the field at the corresponding physical directions,
        then projects the vector field onto the local spherical basis of the child
        frame before applying the requested polarization convention.

        This is useful for reproducing TICRA cuts or grids such as

            spherical_cut(coor_sys : ref(rotated_coordinate_system))

        without explicitly rotating the SWE coefficients.

        Args:
            theta_start_rad:
                Starting colatitude angle in radians in the child coordinate system.
                Negative values are supported, as in TICRA-style cuts.

            theta_end_rad:
                Ending colatitude angle in radians in the child coordinate system.

            ntheta:
                Number of samples along the colatitude direction.

            phi_start_rad:
                Starting azimuth angle in radians in the child coordinate system.

            phi_end_rad:
                Ending azimuth angle in radians in the child coordinate system.

            nphi:
                Number of samples along the azimuth direction.

            frame:
                Either an :class:`EulerAngles` instance or a ``3x3`` matrix mapping
                Cartesian vector components from the child frame to the base frame.
                If an :class:`EulerAngles` instance is provided, its
                ``as_child_to_base_matrix()`` method is used.

            polarization:
                Output polarization basis. If ``Polarization.THETA_PHI``, the
                returned components are expressed in the local spherical basis of
                the child frame. If Ludwig-3 is requested, the projection uses the
                child-frame azimuth angles.

            epsilon:
                Desired accuracy for ``ducc0.sht.synthesis_general``.

            use_ticra_phase:
                If ``True``, complex-conjugate the returned components to match the
                TICRA GRASP convention used in ``.cut`` and ``.grd`` files.

        Returns:
            Tuple ``(comp1, comp2)`` of complex arrays with shape ``(ntheta, nphi)``.
        """
        theta = np.linspace(theta_start_rad, theta_end_rad, num=ntheta)
        phi = np.linspace(phi_start_rad, phi_end_rad, num=nphi)

        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        rotation_matrix_child_to_base = self._coor_sys_to_rotation_matrix(frame)

        return self.evaluate_in_frame(
            theta_rad=theta_grid,
            phi_rad=phi_grid,
            rotation_matrix_child_to_base=rotation_matrix_child_to_base,
            polarization=polarization,
            epsilon=epsilon,
            use_ticra_phase=use_ticra_phase,
        )

    def evaluate_cut_in_frame(
        self,
        phi_angle_rad: float,
        theta_start_rad: float,
        theta_end_rad: float,
        ntheta: int,
        frame: EulerAngles | np.ndarray,
        polarization: Polarization,
        epsilon: float = 1e-8,
        use_ticra_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a 1D spherical cut defined in a rotated coordinate system.

        This is the frame-aware counterpart of :meth:`evaluate_cut`. The cut is
        parameterized by a constant child-frame azimuth ``phi_angle_rad`` and a
        range of child-frame colatitude angles. The field coefficients themselves
        are not rotated. Instead, the requested child-frame directions are mapped
        into the base frame, the field is evaluated there, and the vector components
        are projected back onto the child-frame spherical basis.

        This method is the appropriate way to reproduce TICRA operations of the form

            spherical_cut(
                coor_sys : ref(rotated_coordinate_system),
                ...
            )

        when the SWE coefficients are expressed in the base coordinate system.

        Args:
            phi_angle_rad:
                Constant azimuth angle of the cut in the child coordinate system.

            theta_start_rad:
                Starting colatitude angle in radians in the child coordinate system.
                Negative values are supported, as in TICRA-style cuts.

            theta_end_rad:
                Ending colatitude angle in radians in the child coordinate system.

            ntheta:
                Number of samples along the cut.

            frame:
                Either an :class:`EulerAngles` instance or a ``3x3`` matrix mapping
                Cartesian vector components from the child frame to the base frame.
                If an :class:`EulerAngles` instance is provided, its
                ``as_child_to_base_matrix()`` method is used.

            polarization:
                Output polarization basis. If ``Polarization.THETA_PHI``, the
                returned components are expressed in the child-frame spherical basis.
                If Ludwig-3 is requested, the projection uses the child-frame
                azimuth ``phi_angle_rad``.

            epsilon:
                Desired accuracy for ``ducc0.sht.synthesis_general``.

            use_ticra_phase:
                If ``True``, complex-conjugate the returned components to match the
                TICRA GRASP convention used in ``.cut`` and ``.grd`` files.

        Returns:
            Tuple ``(comp1, comp2)`` of complex arrays with shape ``(ntheta,)``.
        """
        e1, e2 = self.evaluate_theta_phi_grid_in_frame(
            theta_start_rad=theta_start_rad,
            theta_end_rad=theta_end_rad,
            ntheta=ntheta,
            phi_start_rad=phi_angle_rad,
            phi_end_rad=phi_angle_rad,
            nphi=1,
            frame=frame,
            polarization=polarization,
            epsilon=epsilon,
            use_ticra_phase=use_ticra_phase,
        )

        return e1.flatten(), e2.flatten()

    def evaluate_theta_phi_grid_in_coordinate_system(
        self,
        theta_start_rad: float,
        theta_end_rad: float,
        ntheta: int,
        phi_start_rad: float,
        phi_end_rad: float,
        nphi: int,
        coor_sys: CoordinateSystem,
        polarization: Polarization,
        epsilon: float = 1e-8,
        use_ticra_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the electric field on an angular grid in a translated and rotated
        coordinate system.

        This is the TICRA-like counterpart of :meth:`evaluate_theta_phi_grid`.
        The coordinate system is described by a :class:`CoordinateSystem`, whose
        origin and Euler angles are interpreted as in TICRA:

        - ``coor_sys.origin_m`` is the position of the child origin expressed in
          the base frame.
        - ``coor_sys.angles`` describes the orientation of the child axes relative
          to the base axes.

        The SWE coefficients stored in this object are assumed to be expressed in
        the base coordinate system.

        To evaluate the field as seen from a child coordinate system whose origin
        is displaced by ``+origin_m``, SWEaver shifts the field phase center by
        ``-origin_m`` and then evaluates the field in the rotated child frame. This
        sign convention reproduces TICRA ``coor_sys_euler_angles(origin: ...)``
        spherical cuts.

        Args:
            theta_start_rad, theta_end_rad:
                Start and end colatitude angles in the child coordinate system.

            ntheta:
                Number of colatitude samples.

            phi_start_rad, phi_end_rad:
                Start and end azimuth angles in the child coordinate system.

            nphi:
                Number of azimuth samples.

            coor_sys:
                Coordinate system in which the angular grid is defined.

            polarization:
                Output polarization basis.

            epsilon:
                Desired accuracy for ``ducc0.sht.synthesis_general``.

            use_ticra_phase:
                If ``True``, complex-conjugate the returned components to match
                TICRA ``.cut`` and ``.grd`` files.

        Returns:
            Tuple of arrays with shape ``(ntheta, nphi)``.
        """
        shifted_field = self.translate_phase_center(*(-coor_sys.origin_m))

        return shifted_field.evaluate_theta_phi_grid_in_frame(
            theta_start_rad=theta_start_rad,
            theta_end_rad=theta_end_rad,
            ntheta=ntheta,
            phi_start_rad=phi_start_rad,
            phi_end_rad=phi_end_rad,
            nphi=nphi,
            frame=coor_sys.angles,
            polarization=polarization,
            epsilon=epsilon,
            use_ticra_phase=use_ticra_phase,
        )

    def evaluate_cut_in_coordinate_system(
        self,
        phi_angle_rad: float,
        theta_start_rad: float,
        theta_end_rad: float,
        ntheta: int,
        coor_sys: CoordinateSystem,
        polarization: Polarization,
        epsilon: float = 1e-8,
        use_ticra_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a 1D spherical cut in a translated and rotated coordinate system.

        This is the TICRA-like counterpart of :meth:`evaluate_cut`. It reproduces
        operations of the form

            spherical_cut(coor_sys : ref(some_csy))

        where ``some_csy`` may include both an ``origin`` and Euler angles.

        The field coefficients are assumed to be expressed in the base coordinate
        system. The angular samples are interpreted in ``coor_sys``. Internally,
        SWEaver shifts the field phase center by ``-coor_sys.origin_m`` and then
        evaluates the cut in the rotated child frame.

        Args:
            phi_angle_rad:
                Constant azimuth angle of the cut in ``coor_sys``.

            theta_start_rad, theta_end_rad:
                Start and end colatitude angles in ``coor_sys``.

            ntheta:
                Number of cut samples.

            coor_sys:
                Coordinate system defining both the cut origin and orientation.

            polarization:
                Output polarization basis.

            epsilon:
                Desired accuracy for ``ducc0.sht.synthesis_general``.

            use_ticra_phase:
                If ``True``, complex-conjugate the returned components to match
                TICRA ``.cut`` files.

        Returns:
            Tuple of 1D arrays with shape ``(ntheta,)``.
        """
        e1, e2 = self.evaluate_theta_phi_grid_in_coordinate_system(
            theta_start_rad=theta_start_rad,
            theta_end_rad=theta_end_rad,
            ntheta=ntheta,
            phi_start_rad=phi_angle_rad,
            phi_end_rad=phi_angle_rad,
            nphi=1,
            coor_sys=coor_sys,
            polarization=polarization,
            epsilon=epsilon,
            use_ticra_phase=use_ticra_phase,
        )

        return e1.flatten(), e2.flatten()

    def to_texture(
        self,
        shape: tuple[int, int] = (512, 1024),
        mode: str | MapCallable = MapMode.intensity,
        polarization: Polarization = Polarization.THETA_PHI,
    ) -> np.ndarray:
        """
        Generate an equirectangular projection of the electric field.

        Convert the spherical harmonic representation in a scalar 2D grid,
        sampling the field over the whole sphere and applying a mapping
        function, e.g., intensity, phase, dB.

        The grid covers the whole interval:
        - ϑ (elevation): from 0 to π (North → South)
        - φ (azimut): from 0 to 2π

        Args:
            shape (tuple[int, int], optional): Texture resolution ``(n_theta, n_phi)``.
            mode (str | MapMode | Callable): Define how to transform the
                complex components of the field (:math:`E_\\theta, E_\\phi`) into
                scalar values. It can either be a member of :class:`MapMode`,
                a string (``db``, ``phase_theta``, etc.), or a custom function
                that accepts the two NumPy arrays ``e_theta`` and ``e_phi`` and
                returns a NumPy array.
            polarization (Polarization, optional): The polarization basis to
                use to calculate the field.

        Returns:
            np.ndarray: 2D array of ``float64`` with size `shape`.

        Example:
            .. code-block:: python

                # Get a representation of the intensity of the field in dB
                texture_db = efield.to_texture(shape=(400, 800), mode="db")

                # Use a custom function to map the data to a scalar
                my_map = lambda et, ep: np.abs(et) / (np.abs(ep) + 1e-10)
                ratio_texture = efield.to_texture(mode=my_map)
        """
        n_theta, n_phi = shape
        e_theta, e_phi = self.evaluate_theta_phi_grid(
            0, np.pi, n_theta, 0, 2 * np.pi, n_phi, polarization
        )

        map_func: MapCallable

        if isinstance(mode, str):
            if mode in MapMode._REGISTRY:
                map_func = MapMode._REGISTRY[mode]
            else:
                available = ", ".join(f"'{m}'" for m in MapMode.list_modes())
                raise ValueError(
                    f"Invalid mode '{mode}'. Available modes are: {available}. "
                    "You can also pass a custom callable."
                )
        else:
            map_func = mode

        return map_func(e_theta, e_phi)

    def show_3d(
        self,
        shape: tuple[int, int] = (300, 600),
        mode: str | Callable = MapMode.intensity,
        polarization: Polarization = Polarization.THETA_PHI,
    ):
        """
        Render an interactive 3D visualization of the electric field on a sphere.

        This method projects the electric field components onto a spherical mesh
        and opens an interactive Plotly session. It is optimized for Jupyter
        environments to allow real-time rotation, zooming, and inspection of
        beam features like sidelobes and phase patterns.

        Args:
            shape (tuple[int, int], optional): The resolution in pixels of the spherical mesh as ``(n_theta, n_phi)``.
                Higher values increase detail but may impact rendering performance.
            mode (str | MapMode | Callable, optional): The mapping function used to convert complex field components
                (:math:`E_\\theta, E_\\phi`) into scalar values. Accepts:

                - A `MapMode` enum member (e.g., `MapMode.DB`).
                - A string key (e.g., "db", "phase_theta").
                - A custom callable: `f(e_theta, e_phi) -> scalar_array`.

            polarization (Polarization, optional): The polarization basis used to evaluate the field (e.g., THETA_PHI,
                LUDWIG3_X).

        Returns:
            plotly.graph_objects.Figure: An interactive Plotly Figure object. In Jupyter environments,
                returning this object will render the widget in the cell output.

        Notes:
            - This method requires `plotly` to be installed.
            - If ripples or high-frequency features are not visible, try increasing
              the `shape` resolution to improve the sampling density of the mesh.
            - To ensure proper rendering in JupyterLab, a kernel restart might be
              required if WebGL context issues occur.

        Example:
            .. code-block:: python

                # Visualize the beam intensity in dB
                efield.show_3d(mode="db")

                # Inspect the phase of the Ludwig-3 X-polarized component
                from sweaver import MapMode, Polarization
                efield.show_3d(mode=MapMode.PHASE_THETA, polarization=Polarization.LUDWIG3_X)
        """
        try:
            import plotly.graph_objects as go  # ty: ignore[unresolved-import]
        except ImportError:
            print(
                "Plotly not found. Install it using 'uv add --group visualization plotly'"
            )
            return

        data = self.to_texture(shape=shape, mode=mode, polarization=polarization)

        # Generate the sphere
        theta = np.linspace(0, np.pi, shape[0])
        phi = np.linspace(0, 2 * np.pi, shape[1])
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        X = np.sin(THETA) * np.cos(PHI)
        Y = np.sin(THETA) * np.sin(PHI)
        Z = np.cos(THETA)

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    surfacecolor=data,
                    colorscale="Viridis",
                    colorbar=dict(title=str(mode)),
                )
            ]
        )

        # Make the box a cube, so that the sphere doesn’t look like an ellipsoid
        fig.update_layout(
            title=f"SWEaver 3D: {str(mode)}",
            scene=dict(
                aspectmode="data",  # This keeps the sphere spherical
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        return fig.show()

    def find_peak(
        self,
        region_theta_rad: tuple[float, float, int] = (0, np.radians(10), 30),
        region_phi_rad: tuple[float, float, int] = (0, np.radians(360), 60),
    ) -> tuple[float, float, float]:
        """
        Finds the direction of maximum intensity and its polarization twist.

        Performs a grid search followed by an optimization to locate the
        peak of the beam. It then evaluates the Ludwig-3 cross-polarization
        ratio at the peak to determine the polarization orientation.

        Args:
            region_theta_rad (tuple[float, float, int], optional): The search region for the elevation angle as (start, end, samples).
            region_phi_rad (tuple[float, float, int], optional): The search region for the azimuthal angle as (start, end, samples).

        Returns:
            tuple[float, float, float]: The coordinates of the peak and its polarization twist in radians:
                (theta_peak, phi_peak, psi_pol).
        """

        # Sample the region where the maximum is
        theta_start_rad, theta_end_rad, ntheta = region_theta_rad
        phi_start_rad, phi_end_rad, nphi = region_phi_rad
        e_co, e_cx = self.evaluate_theta_phi_grid(
            theta_start_rad=theta_start_rad,
            theta_end_rad=theta_end_rad,
            ntheta=ntheta,
            phi_start_rad=phi_start_rad,
            phi_end_rad=phi_end_rad,
            nphi=nphi,
            polarization=Polarization.LUDWIG3_X,
        )

        intensity = np.abs(e_co) ** 2 + np.abs(e_cx) ** 2

        # Find the rough position of the maximum
        region_theta = np.linspace(theta_start_rad, theta_end_rad, ntheta)
        region_phi = np.linspace(phi_start_rad, phi_end_rad, nphi)

        idx_max = np.argmax(intensity)
        theta_max_idx, phi_max_idx = np.unravel_index(idx_max, intensity.shape)
        theta_0 = region_theta[theta_max_idx]
        phi_0 = region_phi[phi_max_idx]

        # Use SciPy to find the accurate position of the maximum
        def objective(coords: tuple[np.float64, np.float64]) -> np.float64:
            cur_theta, cur_phi = coords
            new_theta, new_phi = self.evaluate_theta_phi_grid(
                theta_start_rad=cur_theta,
                theta_end_rad=cur_theta,
                ntheta=1,
                phi_start_rad=cur_phi,
                phi_end_rad=cur_phi,
                nphi=1,
                polarization=Polarization.LUDWIG3_X,
            )
            return -(np.abs(new_theta[0]) ** 2 + np.abs(new_phi[0]) ** 2)

        res = scipy.optimize.minimize(
            objective,
            x0=(theta_0, phi_0),
            bounds=[(theta_start_rad, theta_end_rad), (phi_start_rad, phi_end_rad)],
            method="Powell",
        )
        theta_peak, phi_peak = res.x

        # Create a copy of this field and rotate it so that the maximum is
        # aligned with +Z
        reoriented_beam = self.rotate_euler(
            EulerAngles(alpha_rad=0.0, beta_rad=-theta_peak, gamma_rad=-phi_peak),
        )

        # Align the polarization axis with the x axis
        # (The polarization axis is the direction of the copolar component of the
        # electric field)

        e_co, e_cx = reoriented_beam.evaluate_theta_phi_grid(
            theta_start_rad=0.0,
            theta_end_rad=0.0,
            ntheta=1,
            phi_start_rad=0.0,
            phi_end_rad=0.0,
            nphi=1,
            polarization=Polarization.LUDWIG3_X,
        )
        psi_pol = float(np.angle(e_co[0, 0] + 1j * e_cx[0, 0]))

        return float(theta_peak), float(phi_peak), psi_pol

    def get_alignment_angles(
        self,
        region_theta_rad: tuple[float, float, int] = (0, np.radians(10), 30),
        region_phi_rad: tuple[float, float, int] = (0, np.radians(360), 60),
    ) -> EulerAngles:
        """
        Compute the Euler angles required to center the beam and align its polarization.

        This function finds the beam's peak and calculates the inverse Z-Y-Z
        Euler rotation needed to bring the peak to the +Z axis and align the
        copolar direction with the +X axis.

        Returns:
            dict[str, float]: A dictionary containing the keys `psi_rad`, `theta_rad`, and `phi_rad`.
                This can be unpacked directly into the :meth:`.rotate` method.

        Example:
            .. code-block:: python

                angles = efield.get_alignment_angles()
                aligned_efield = efield.rotate(**angles)
        """
        theta, phi, psi = self.find_peak(region_theta_rad, region_phi_rad)

        # The inverse of a beam at (theta, phi) with twist (psi)
        return EulerAngles(alpha_rad=-psi, beta_rad=-theta, gamma_rad=-phi)

    def align(
        self,
        region_theta_rad: tuple[float, float, int] = (0, np.radians(10), 30),
        region_phi_rad: tuple[float, float, int] = (0, np.radians(360), 60),
    ) -> "ElectricField":
        """Convenience method that finds the peak and returns a re-aligned copy."""
        angles = self.get_alignment_angles(region_theta_rad, region_phi_rad)
        return self.rotate_euler(angles)


@dataclass
class Beam:
    """
    A beam pattern decomposed into Stokes parameters (I, Q, U) via spherical harmonics
    (Spin-0 and Spin-2).

    Unlike :class:`.ElectricField` which uses physical components
    (:math:`E_\\theta, E_\\phi`), this class provides a representation suitable for
    CMB data analysis and beam convolution libraries. The field is described
    using the standard CMB convention:

    - :math:`a_{\\ell m}^I`: Spin-0 harmonic coefficients representing the total intensity (Stokes I).

    - :math:`a_{\\ell m}^E`: Spin-2 harmonic coefficients (E-mode) representing gradient-like polarization.

    - :math:`a_{\\ell m}^B`: Spin-2 harmonic coefficients (B-mode) representing curl-like polarization.
    """

    alm_i: np.ndarray
    """1D array of Stokes I harmonic coefficients (:math:`a_{\\ell m}^I`)."""

    alm_e: np.ndarray
    """1D array of Stokes Q/U E-mode harmonic coefficients (:math:`a_{\\ell m}^E`)."""

    alm_b: np.ndarray
    """1D array of Stokes Q/U B-mode harmonic coefficients (:math:`a_{\\ell m}^B`)."""

    lmax: int
    """Maximum multipole order (:math:`\\ell`) for the spherical harmonic expansion."""

    mmax: int
    """Maximum azimuthal order (:math:`m`) for the spherical harmonic expansion."""

    frequency_ghz: float | None = None
    """The frequency of the beam in GHz, if known."""

    @classmethod
    def from_electric_field(
        cls,
        electric_field: ElectricField,
        lmax: int | None = None,
        mmax: int | None = None,
    ) -> "Beam":
        """
        Convert an `ElectricField` object into a `Beam` object.

        This method projects the :math:`E_\\theta` and :math:`E_\\phi` components over a spatial
        Gauss-Legendre grid, computes the local Stokes parameters (:math:`I, Q, U`),
        and then performs a Spin-0 and Spin-2 spherical harmonic transform to
        extract the :math:`I, E, B` coefficients.

        Args:
            electric_field (ElectricField): The input electric field object.
            lmax (int | None, optional): The maximum multipole order :math:`\\ell` to compute.
                If `None`, it defaults to the `lmax` of the input field.
            mmax (int | None, optional): The maximum azimuthal order :math:`m` to compute.
                If `None`, it defaults to `lmax`.

        Returns:
            Beam: A new instance populated with the computed harmonic coefficients.
        """
        E_theta, E_phi = electric_field.project_to_gl()
        assert E_theta.shape == E_phi.shape

        stokes_I = np.array(np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2, dtype=np.float64)
        stokes_Q = np.array(np.abs(E_theta) ** 2 - np.abs(E_phi) ** 2, dtype=np.float64)
        stokes_U = np.array(2 * np.real(E_theta * np.conj(E_phi)), dtype=np.float64)

        if lmax is None:
            lmax = electric_field.lmax
        if mmax is None:
            mmax = lmax

        n_theta, n_phi = E_theta.shape

        # Get back the b_ℓm
        alm_I = ducc0.sht.analysis_2d(
            map=np.ascontiguousarray(stokes_I.reshape(1, n_theta, n_phi)),
            spin=0,
            geometry="GL",
            lmax=lmax,
        )

        alm_pol = ducc0.sht.analysis_2d(
            map=np.ascontiguousarray([stokes_Q, stokes_U]),
            spin=2,
            geometry="GL",
            lmax=lmax,
        )

        return cls(
            alm_i=alm_I[0],
            alm_e=alm_pol[0],
            alm_b=alm_pol[1],
            lmax=lmax,
            mmax=mmax,
        )

    def get_idx(self, ell: int, m: int) -> int:
        """
        Return the index of an :math:`a_{\\ell m}` coefficient given a specific (:math:`\\ell, m`) pair.

        Note that only coefficients with :math:`m \\geq 0` are stored in the object,
        so the function will raise an ``AssertionError`` if :math:`m < 0`.

        Args:
            ell (int): Multipole order (:math:`\\ell`).
            m (int): Azimuthal order (:math:`m \\geq 0`).

        Returns:
            int: The zero-based index in the underlying coefficient arrays (`alm_i`, etc.).
        """
        assert ell >= 0, "ℓ={ell} cannot be negative"
        assert m >= 0, "m={m} cannot be negative"
        assert m <= ell, f"{m=} > {ell=} cannot be greater than ℓ={ell}"
        return m * (2 * self.lmax + 1 - m) // 2 + ell

    def get_alms(self, ell: int, m: int) -> tuple[complex, complex, complex]:
        """
        Retrieve the harmonic coefficients (:math:`a_{\\ell m}^I, a_{\\ell m}^E, a_{\\ell m}^B`)
        for a given pair (:math:`\\ell, m`).

        This method supports both positive and negative values of :math:`m`. When :math:`m < 0`,
        the coefficients are conjugated and the phase symmetry factor :math:`(-1)^m` is applied.

        Args:
            ell (int): Multipole order :math:`\\ell`.
            m (int): Azimuthal order :math:`m`. Can be negative.

        Returns:
            tuple[complex, complex, complex]: A 3-element tuple containing the harmonic
                coefficients for Stokes I (Spin-0), E-mode (Spin-2), and B-mode (Spin-2).

        Raises:
            ValueError: If :math:`\\ell` is out of bounds (:math:`\\ell < 0` or :math:`\\ell > lmax`).
        """
        if not (0 <= ell <= self.lmax):
            raise ValueError(f"out-of-bounds ℓ={ell}")

        if abs(m) > min(ell, self.mmax):
            return 0j, 0j, 0j

        idx = self.get_idx(ell, abs(m))

        val_i = self.alm_i[idx]
        val_e = self.alm_e[idx]
        val_b = self.alm_b[idx]

        if m < 0:
            phase = (-1) ** abs(m)
            val_i = phase * np.conj(val_i)
            val_e = phase * np.conj(val_e)
            val_b = phase * np.conj(val_b)

        return val_i, val_e, val_b

    def angular_power_spectra(
        self, ell_start: int = 2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the angular power spectra (:math:`C_\\ell`) for I, E, and B modes.

        The angular power spectrum describes the variance of the coefficients at each
        multipole :math:`\\ell`, summing over all valid :math:`m`.

        Args:
            ell_start (int, optional): The first multipole to compute. Defaults to 2,
                which is typically the lowest meaningful order for polarized beams.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A 4-element tuple
                containing 1D arrays:

                - `ells`: The multipole sequence (from `ell_start` to `lmax`).
                - `C_ℓ^I`: Power spectrum of intensity.
                - `C_ℓ^E`: Power spectrum of E-mode polarization.
                - `C_ℓ^B`: Power spectrum of B-mode polarization.
        """
        ells = np.arange(ell_start, self.lmax + 1)
        cl_i = np.zeros_like(ells, dtype=float)
        cl_e = np.zeros_like(ells, dtype=float)
        cl_b = np.zeros_like(ells, dtype=float)

        for i, ell in enumerate(ells):
            # 1. Termine m = 0 (nessuna simmetria, contato una volta sola)
            idx_0 = self.get_idx(ell, 0)
            sum_i = np.abs(self.alm_i[idx_0]) ** 2
            sum_e = np.abs(self.alm_e[idx_0]) ** 2
            sum_b = np.abs(self.alm_b[idx_0]) ** 2

            # 2. Termini m > 0 (contati due volte per riflettere anche m < 0)
            for m in range(1, min(ell, self.mmax) + 1):
                idx = self.get_idx(ell, m)
                sum_i += 2 * (np.abs(self.alm_i[idx]) ** 2)
                sum_e += 2 * (np.abs(self.alm_e[idx]) ** 2)
                sum_b += 2 * (np.abs(self.alm_b[idx]) ** 2)

            # 3. Normalizzazione
            cl_i[i] = sum_i / (2 * ell + 1)
            cl_e[i] = sum_e / (2 * ell + 1)
            cl_b[i] = sum_b / (2 * ell + 1)

        return ells, cl_i, cl_e, cl_b


def read_sph_electric_field(
    f: TextIO | str | Path,
    frequency_idx: int = 0,
) -> ElectricField:
    """Read the SWE of an electric field at a specified frequency from a GRASP .sph file.

    This is a convenience function that wraps :func:`read_sph_frequency_block`.

    Args:
        f (TextIO | str | Path): The file to read from. It can be a path
            (either a string or a ``pathlib.Path`` object) or a file-like
            object opened in text mode. If a path is provided, the function
            will automatically handle GZip-compressed files.
        frequency_idx (int): The 0-based index of the frequency block to
            read.

    Returns:
        ElectricField: The parsed electric field.
    """
    freq_block = read_sph_frequency_block(f, frequency_idx)
    return ElectricField.from_frequency_block(freq_block)
