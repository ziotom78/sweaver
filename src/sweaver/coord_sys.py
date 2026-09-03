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

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def _nearest_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Return the closest proper rotation matrix to ``matrix``.

    This is useful when a rotation matrix is reconstructed from printed TICRA
    axes, which may not be exactly orthonormal because of finite precision.
    """
    u, _, vt = np.linalg.svd(matrix)
    r = u @ vt

    # Enforce det(R) = +1, not -1.
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt

    return r


@dataclass
class EulerAngles:
    """
    A collection of three floating-point values representing three
    (intrinsic) Euler angles, in radians:
        - alpha_rad (float): Rotation around Z axis (first).
        - beta_rad (float): Rotation around new Y axis.
        - gamma_rad (float): Rotation around new Z axis (last).
    """

    alpha_rad: float = 0.0
    beta_rad: float = 0.0
    gamma_rad: float = 0.0

    def inverse(self) -> "EulerAngles":
        """
        Return the set of Euler angles that represent the inverse rotation
        with respect to `self`.
        """

        return EulerAngles(
            alpha_rad=-self.gamma_rad,
            beta_rad=-self.beta_rad,
            gamma_rad=-self.alpha_rad,
        )

    def as_child_to_base_matrix(self) -> np.ndarray:
        """
        Return the rotation matrix mapping Cartesian vector components from the
        child coordinate system to the base coordinate system.

        The convention is

            v_base = R_child_to_base @ v_child

        with

            R_child_to_base = Rz(alpha) @ Ry(beta) @ Rz(gamma).

        The child z-axis therefore points in the base frame toward

            theta = beta
            phi   = alpha

        while gamma is a twist around the child z-axis.
        """
        return (
            _rotation_matrix_z(self.alpha_rad)
            @ _rotation_matrix_y(self.beta_rad)
            @ _rotation_matrix_z(self.gamma_rad)
        )

    def as_base_to_child_matrix(self) -> np.ndarray:
        """
        Return the inverse rotation matrix, mapping base-frame components to
        child-frame components.
        """
        return self.as_child_to_base_matrix().T

    def __str__(self):
        return f"EulerAngles(α={np.rad2deg(self.alpha_rad)}°, β={np.rad2deg(self.beta_rad)}°, γ={np.rad2deg(self.gamma_rad)}°)"


def _euler_from_child_to_base_matrix(
    rotation_matrix_child_to_base: np.ndarray,
    *,
    atol: float = 1e-12,
) -> EulerAngles:
    """
    Convert a child-to-base rotation matrix to Z-Y-Z Euler angles.

    The convention is

        R = Rz(alpha) @ Ry(beta) @ Rz(gamma)

    where ``R`` maps Cartesian vector components from the child coordinate
    system to the base coordinate system.

    The returned Euler-angle representation is not unique in gimbal-lock cases,
    but the reconstructed matrix is equivalent.
    """
    r = np.asarray(rotation_matrix_child_to_base, dtype=float)

    if r.shape != (3, 3):
        raise ValueError(
            f"rotation_matrix_child_to_base must have shape (3, 3), got {r.shape}"
        )

    # Make the conversion robust against tiny non-orthogonality from printed axes.
    r = _nearest_rotation_matrix(r)

    det = np.linalg.det(r)
    if not np.isclose(det, 1.0, atol=atol):
        raise ValueError(
            "rotation_matrix_child_to_base must be a proper rotation matrix; "
            f"determinant is {det}"
        )

    beta = np.arctan2(np.hypot(r[0, 2], r[1, 2]), r[2, 2])
    sin_beta = np.sin(beta)

    if abs(sin_beta) > atol:
        alpha = np.arctan2(r[1, 2], r[0, 2])
        gamma = np.arctan2(r[2, 1], -r[2, 0])
    else:
        # Gimbal lock. alpha and gamma are not separately determined.
        #
        # For beta ≈ 0:
        #     R ≈ Rz(alpha + gamma)
        #
        # We choose gamma = 0 and put the whole z-rotation into alpha.
        if r[2, 2] > 0.0:
            beta = 0.0
            alpha = np.arctan2(r[1, 0], r[0, 0])
            gamma = 0.0

        # For beta ≈ pi:
        #     R ≈ Rz(alpha) @ Ry(pi) @ Rz(gamma)
        #
        # alpha and gamma are again degenerate. We choose gamma = 0.
        else:
            beta = np.pi
            alpha = np.arctan2(-r[1, 0], -r[0, 0])
            gamma = 0.0

    return EulerAngles(
        alpha_rad=float(alpha),
        beta_rad=float(beta),
        gamma_rad=float(gamma),
    )


@dataclass(frozen=True)
class CoordinateSystem:
    """
    Coordinate system defined relative to a base frame.

    This class mirrors the information used by TICRA coordinate-system objects:
    a translation of the coordinate-system origin and a rotation of the
    coordinate-system axes relative to a base frame.

    The convention is TICRA-like:

        origin_m

    is the position of the child coordinate-system origin expressed in the base
    coordinate system, in meters.

    The rotation is represented by ``angles``. The corresponding matrix maps
    Cartesian vector components from the child frame to the base frame:

        v_base = R_child_to_base @ v_child

    where

        R_child_to_base = Rz(alpha) @ Ry(beta) @ Rz(gamma)

    A field represented by SWE coefficients in the base frame can be evaluated
    in this coordinate system by shifting the field phase center by
    ``-origin_m`` and then evaluating directions in the rotated child frame.
    """

    origin_m: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0]),
    )
    angles: EulerAngles = field(
        default_factory=lambda: EulerAngles(
            alpha_rad=0.0,
            beta_rad=0.0,
            gamma_rad=0.0,
        )
    )

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_m, dtype=float)

        if origin.shape != (3,):
            raise ValueError(f"origin_m must have shape (3,), got {origin.shape}")

        object.__setattr__(self, "origin_m", origin)

    @classmethod
    def identity(cls) -> "CoordinateSystem":
        """Return the base coordinate system."""
        return cls()

    @classmethod
    def from_ticra_degrees(
        cls,
        origin_m: np.ndarray | None = None,
        alpha_deg: float = 0.0,
        beta_deg: float = 0.0,
        gamma_deg: float = 0.0,
    ) -> "CoordinateSystem":
        """
        Build a coordinate system from TICRA-style Euler angles in degrees.
        """
        if origin_m is None:
            origin_m = np.zeros(3)

        return cls(
            origin_m=origin_m,
            angles=EulerAngles(
                alpha_rad=np.deg2rad(alpha_deg),
                beta_rad=np.deg2rad(beta_deg),
                gamma_rad=np.deg2rad(gamma_deg),
            ),
        )

    @classmethod
    def from_axes(
        cls,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        origin_m: np.ndarray | None = None,
    ) -> "CoordinateSystem":
        """
        Build a coordinate system from TICRA-style x_axis and y_axis definitions.

        TICRA ``coor_sys`` objects define the child-frame x and y axes expressed
        in the base coordinate system. This constructor reconstructs an orthonormal
        right-handed frame and converts it to the internal Euler-angle
        representation.

        Args:
            x_axis:
                Child x-axis expressed in the base frame.

            y_axis:
                Child y-axis expressed in the base frame.

            origin_m:
                Position of the child origin expressed in the base frame, in meters.
                If omitted, the origin is zero.

        Returns:
            CoordinateSystem.
        """
        x = np.asarray(x_axis, dtype=float)
        y = np.asarray(y_axis, dtype=float)

        if x.shape != (3,):
            raise ValueError(f"x_axis must have shape (3,), got {x.shape}")

        if y.shape != (3,):
            raise ValueError(f"y_axis must have shape (3,), got {y.shape}")

        x = x / np.linalg.norm(x)

        # Gram-Schmidt: remove any small component of y along x.
        y = y - np.dot(y, x) * x
        y = y / np.linalg.norm(y)

        z = np.cross(x, y)

        rotation_matrix_child_to_base = np.column_stack([x, y, z])

        return cls.from_matrix(
            rotation_matrix_child_to_base=rotation_matrix_child_to_base,
            origin_m=origin_m,
            orthonormalize=True,
        )

    def as_child_to_base_matrix(self) -> np.ndarray:
        """
        Return the matrix mapping Cartesian vectors from child to base frame.
        """
        return self.angles.as_child_to_base_matrix()

    def as_base_to_child_matrix(self) -> np.ndarray:
        """
        Return the matrix mapping Cartesian vectors from base to child frame.
        """
        return self.angles.as_base_to_child_matrix()

    @property
    def has_translation(self) -> bool:
        """Return True if this coordinate system has a non-zero origin."""
        return bool(np.any(self.origin_m != 0.0))

    def relative_to(self, parent: "CoordinateSystem") -> "CoordinateSystem":
        """
        Return this coordinate system expressed relative to ``parent``.

        Both ``self`` and ``parent`` must be expressed relative to the same external
        base frame.

        If ``self`` represents frame C relative to global frame G, and ``parent``
        represents frame P relative to the same G, this method returns frame C
        expressed relative to P.

        This is useful when a field is represented in frame P but must be evaluated
        on a cut or grid defined in frame C.
        """
        r_parent_to_global = parent.as_child_to_base_matrix()
        r_self_to_global = self.as_child_to_base_matrix()

        r_self_to_parent = r_parent_to_global.T @ r_self_to_global

        origin_self_minus_parent_global = self.origin_m - parent.origin_m
        origin_self_in_parent = r_parent_to_global.T @ origin_self_minus_parent_global

        return CoordinateSystem.from_matrix(
            rotation_matrix_child_to_base=r_self_to_parent,
            origin_m=origin_self_in_parent,
            orthonormalize=True,
        )

    @classmethod
    def from_matrix(
        cls,
        rotation_matrix_child_to_base: np.ndarray,
        origin_m: np.ndarray | None = None,
        *,
        orthonormalize: bool = True,
    ) -> "CoordinateSystem":
        """
        Build a coordinate system from a child-to-base rotation matrix.

        Args:
            rotation_matrix_child_to_base:
                A ``3×3`` matrix mapping Cartesian vector components from the
                child frame to the base frame:

                    v_base = R_child_to_base @ v_child

                The matrix is interpreted using the same convention as
                :meth:`as_child_to_base_matrix`.

            origin_m:
                Position of the child coordinate-system origin expressed in the
                base coordinate system, in meters. If omitted, the origin is
                assumed to coincide with the base origin.

            orthonormalize:
                If ``True``, project the input matrix onto the nearest proper
                rotation matrix before converting it to Euler angles. This is
                useful for matrices reconstructed from printed TICRA axes.

        Returns:
            CoordinateSystem:
                Coordinate system with Euler angles equivalent to the supplied
                rotation matrix.
        """
        r = np.asarray(rotation_matrix_child_to_base, dtype=float)

        if r.shape != (3, 3):
            raise ValueError(
                f"rotation_matrix_child_to_base must have shape (3, 3), got {r.shape}"
            )

        if orthonormalize:
            r = _nearest_rotation_matrix(r)

        angles = _euler_from_child_to_base_matrix(r)

        if origin_m is None:
            origin_m = np.zeros(3)

        return cls(
            origin_m=np.asarray(origin_m, dtype=float),
            angles=angles,
        )


def get_euler_from_ticra_axes(
    x_axis: np.typing.ArrayLike, y_axis: np.typing.ArrayLike
) -> EulerAngles:
    """
    Converts TICRA Cartesian axis definitions into active Z-Y-Z Euler angles.

    Args:
        x_vec: An array of 3 floats representing a normalized vector in the form ``[x, y, z]``
        y_vec: An array of 3 floats representing a normalized vector in the form ``[x, y, z]``

    Returns:
        An object of type :class:`.EulerAngles`.
    """

    x_vec = np.array(x_axis)
    assert x_vec.size == 3, f"The X axis ({x_vec}) must have 3 members"

    y_vec = np.array(y_axis)
    assert y_vec.size == 3, f"The Y axis ({y_vec}) must have 3 members"

    # The Z-axis is the cross product of X and Y
    z_vec = np.cross(x_vec, y_vec)

    # Construct the rotation matrix (columns are the local basis vectors)
    rot_matrix = np.column_stack((x_vec, y_vec, z_vec))

    # Scipy 'ZYZ' (capitalized) represents intrinsic active rotations,
    # which exactly matches the Wigner-D matrix convention in ducc0.
    rot = Rotation.from_matrix(rot_matrix)
    alpha, beta, gamma = rot.as_euler("ZYZ", degrees=False)

    return EulerAngles(alpha_rad=alpha, beta_rad=beta, gamma_rad=gamma)


def get_euler_from_grasp_angles(
    theta_rad: float,
    phi_rad: float,
    psi_rad: float,
) -> EulerAngles:
    """
    Convert TICRA (ϑ, φ, ψ) GRASP angles into active Z-Y-Z Euler angles.

    According to the TICRA Tools manual, the mapping between GRASP angles and
    intrinsic Z-Y-Z Euler angles is: (α, β, γ) = (φ, ϑ, -φ + ψ).

    Args:
        theta_rad (float): The TICRA ϑ angle, in radians
        phi_rad (float): The TICRA φ angle, in radians
        psi_rad (float): The TICRA ψ angle, in radians

    Returns:
        An object of type :class:`.EulerAngles`.
    """
    alpha = phi_rad
    beta = theta_rad
    gamma = -phi_rad + psi_rad

    return EulerAngles(alpha_rad=alpha, beta_rad=beta, gamma_rad=gamma)
