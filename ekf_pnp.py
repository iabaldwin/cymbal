import numpy as np
from scipy.spatial.transform import Rotation
import sympy as sp
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class EKFState:
    """Class to hold EKF state vector and covariance"""
    # State vector: [cx, cy, cz, q0, q1, q2, q3, vx, vy, vz, wx, wy, wz]
    s: np.ndarray  # 13x1 state vector
    P: np.ndarray  # 13x13 covariance matrix

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Return conjugate of quaternion"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def angle_axis_to_quaternion(angle: float, axis: np.ndarray) -> np.ndarray:
    """Convert angle-axis representation to quaternion"""
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    sin_half = np.sin(half_angle)
    return np.array([
        np.cos(half_angle),
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half
    ])

class EKFPnP:
    def __init__(self, focal_length: float, world_points: np.ndarray):
        """
        Initialize EKF-PnP tracker

        Args:
            focal_length: Camera focal length in pixels
            world_points: Nx3 array of 3D world points
        """
        self.f = focal_length
        self.world_points = world_points

        # Process noise parameters
        self.sigma_a = 1.0  # Linear acceleration noise
        self.sigma_alpha = 0.1  # Angular acceleration noise

        # Initialize symbolic Jacobians
        self._init_symbolic_jacobians()

    def _init_symbolic_jacobians(self):
        """Initialize symbolic Jacobians for state transition and measurement"""
        # State variables
        c = sp.Matrix(sp.symbols('cx cy cz'))
        q = sp.Matrix(sp.symbols('q0 q1 q2 q3'))
        v = sp.Matrix(sp.symbols('vx vy vz'))
        w = sp.Matrix(sp.symbols('wx wy wz'))
        dt = sp.Symbol('dt')

        # Process noise
        nu = sp.Matrix(sp.symbols('nu1 nu2 nu3'))
        omega = sp.Matrix(sp.symbols('omega1 omega2 omega3'))

        # State transition function
        f = sp.Matrix([
            c + (v + nu)*dt,
            q,  # Quaternion update handled numerically
            v + nu,
            w + omega
        ])

        # Compute state transition Jacobians
        s = sp.Matrix([*c, *q, *v, *w])
        u = sp.Matrix([*nu, *omega])

        self.F_sym = f.jacobian(s)
        self.G_sym = f.jacobian(u)

        # Measurement model Jacobians
        # For a single point x_w
        x_w = sp.Matrix(sp.symbols('x_w y_w z_w'))

        # Quaternion to rotation matrix
        R = sp.Matrix([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
            [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
            [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]
        ])

        # Transform point to camera frame
        x_c = R * (x_w - c)

        # Project to image plane
        f_sym = sp.Symbol('f')
        x_i = f_sym * sp.Matrix([x_c[0]/x_c[2], x_c[1]/x_c[2]])

        # Compute measurement Jacobian for a single point
        # We only need derivatives w.r.t. position and orientation
        # since velocities don't affect the measurement
        h_syms = [*c, *q]
        H_point = x_i.jacobian(h_syms)

        # Create lambda function for fast evaluation
        self.H_point_func = sp.lambdify(
            [*h_syms, *x_w, f_sym],
            H_point,
            modules='numpy'
        )

    def predict(self, state: EKFState, dt: float) -> EKFState:
        """
        Predict next state using constant velocity motion model

        Args:
            state: Current state
            dt: Time step

        Returns:
            Predicted next state
        """
        # Extract state components
        c = state.s[0:3]
        q = state.s[3:7]
        v = state.s[7:10]
        w = state.s[10:13]

        # Predict translation
        c_pred = c + v*dt

        # Predict rotation using quaternion kinematics
        angle = np.linalg.norm(w) * dt
        if angle > 0:
            axis = w / np.linalg.norm(w)
            dq = angle_axis_to_quaternion(angle, axis)
            q_pred = quaternion_multiply(q, dq)
        else:
            q_pred = q

        # Assemble predicted state
        s_pred = np.concatenate([c_pred, q_pred, v, w])

        # Compute Jacobians using symbolic expressions
        F = np.array(self.F_sym.evalf(subs={
            'cx':c[0], 'cy':c[1], 'cz':c[2],
            'q0':q[0], 'q1':q[1], 'q2':q[2], 'q3':q[3],
            'vx':v[0], 'vy':v[1], 'vz':v[2],
            'wx':w[0], 'wy':w[1], 'wz':w[2],
            'dt':dt
        })).astype(np.float64)

        G = np.array(self.G_sym.evalf(subs={
            'dt':dt
        })).astype(np.float64)

        # Process noise covariance
        Q = np.diag([
            self.sigma_a**2, self.sigma_a**2, self.sigma_a**2,
            self.sigma_alpha**2, self.sigma_alpha**2, self.sigma_alpha**2
        ])

        # Predict covariance
        P_pred = F @ state.P @ F.T + G @ Q @ G.T

        return EKFState(s_pred, P_pred)

    def _measurement_function(self, state: EKFState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute predicted image points and measurement Jacobian

        Args:
            state: Current state

        Returns:
            predicted_points: Nx2 array of predicted image points
            H: 2Nx13 measurement Jacobian
        """
        # Extract state components
        c = state.s[0:3]
        q = state.s[3:7]

        # Convert quaternion to rotation matrix
        R = Rotation.from_quat(q[1:4].tolist() + [q[0]]).as_matrix()

        # Transform points to camera frame
        points_cam = (R @ (self.world_points - c.reshape(3, 1).T).T).T

        # Project to image plane
        predicted_points = self.f * points_cam[:, :2] / points_cam[:, 2:3]

        # Build full measurement Jacobian
        H = np.zeros((2*len(self.world_points), 13))

        for i, world_point in enumerate(self.world_points):
            # Compute Jacobian for this point using symbolic expression
            H_point = self.H_point_func(
                *state.s[0:7],  # Position and quaternion
                *world_point,   # World point
                self.f         # Focal length
            )

            # Fill in the relevant part of H
            H[2*i:2*i+2, 0:7] = H_point
            # Zeros for velocity components since they don't affect measurement

        return predicted_points, H

    def update(self, state: EKFState, image_points: np.ndarray,
               measurement_noise: Optional[np.ndarray] = None) -> EKFState:
        """
        Update state using image measurements

        Args:
            state: Predicted state
            image_points: Nx2 array of observed image points
            measurement_noise: Optional 2Nx2N measurement noise covariance

        Returns:
            Updated state
        """
        if measurement_noise is None:
            # Default measurement noise
            measurement_noise = np.eye(2*len(image_points)) * 1.0

        # Compute predicted measurements and Jacobian
        pred_points, H = self._measurement_function(state)

        # Innovation
        innovation = (image_points - pred_points).reshape(-1)

        # Kalman gain
        S = H @ state.P @ H.T + measurement_noise
        K = state.P @ H.T @ np.linalg.inv(S)

        # Update state
        state_update = K @ innovation
        s_updated = state.s + state_update

        # Normalize quaternion
        s_updated[3:7] = s_updated[3:7] / np.linalg.norm(s_updated[3:7])

        # Update covariance
        P_updated = (np.eye(13) - K @ H) @ state.P

        return EKFState(s_updated, P_updated)
