import unittest
from typing import Tuple
import numpy as np

from scipy.spatial.transform import Rotation
from ekf_pnp import EKFPnP, EKFState

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have Qt installed

import cv2

def create_trajectory(n_frames: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple circular camera trajectory with smooth rotations.

    Args:
        n_frames: Number of frames in trajectory

    Returns:
        positions: (n_frames, 3) array of camera positions
        rotations: (n_frames, 3, 3) array of camera rotation matrices
    """
    # Time parameter
    t = np.linspace(0, 2*np.pi, n_frames)

    # Create simple circular pattern for camera position
    positions = np.zeros((n_frames, 3))

    # Scale factor for the circle
    radius = 2.0
    base_z = 10.0  # Base distance from origin

    # Position trajectory - simple circle in XY plane
    positions[:, 0] = radius * np.cos(t)  # X motion
    positions[:, 1] = radius * np.sin(t)  # Y motion
    positions[:, 2] = base_z * np.ones_like(t)  # Constant Z

    # Create smooth rotation trajectory
    rotations = np.zeros((n_frames, 3, 3))

    for i in range(n_frames):
        # Always look at origin
        position = positions[i]
        forward = -position  # Look toward origin
        forward = forward / np.linalg.norm(forward)

        # Use world up vector
        world_up = np.array([0, 0, 1])

        # Compute right vector
        right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)

        # Compute up vector
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # Create rotation matrix (camera frame to world frame)
        R = np.column_stack([right, up, -forward])

        # Ensure the matrix is orthonormal with determinant +1
        U, _, Vh = np.linalg.svd(R)
        R = U @ Vh  # Note: removed the intermediate transpose
        if np.linalg.det(R) < 0:  # If determinant is -1, flip it to +1
            R = -R

        rotations[i] = R

    return positions, rotations

def test_trajectory():
    """Unit test for trajectory generation"""
    n_frames = 100
    positions, rotations = create_trajectory(n_frames)

    # Test shapes
    assert positions.shape == (n_frames, 3), f"Wrong position shape: {positions.shape}"
    assert rotations.shape == (n_frames, 3, 3), f"Wrong rotation shape: {rotations.shape}"

    # Test rotation matrix properties
    for i in range(n_frames):
        R = rotations[i]
        # Test orthogonality
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), f"Non-orthogonal rotation at frame {i}"
        # Test determinant
        assert np.abs(np.linalg.det(R) - 1) < 1e-6, f"Non-unit determinant at frame {i}"

    # Test motion smoothness
    pos_diff = np.diff(positions, axis=0)
    pos_speed = np.linalg.norm(pos_diff, axis=1)
    assert np.all(pos_speed < 1.0), "Position changes too abrupt"

    # Test rotation smoothness with more lenient threshold
    max_angle = 0  # Track maximum angle for debugging
    for i in range(1, n_frames):
        R1 = rotations[i-1]
        R2 = rotations[i]
        # Compute angular difference
        dR = R2 @ R1.T
        angle = np.arccos((np.trace(dR) - 1) / 2)
        max_angle = max(max_angle, angle)
        # Increase threshold to 0.2 radians (≈11.5 degrees)
        if angle >= 0.2:  # More lenient threshold
            print(f"Large rotation at frame {i}:")
            print(f"Angle (degrees): {np.degrees(angle)}")
            print(f"Previous rotation:\n{R1}")
            print(f"Current rotation:\n{R2}")
        assert angle < 0.2, f"Rotation change too large at frame {i}: {np.degrees(angle)} degrees"

    print(f"Maximum rotation angle between frames: {np.degrees(max_angle):.2f} degrees")
    return positions, rotations

def generate_world_points(n_points: int = 20) -> np.ndarray:
    """
    Generate synthetic 3D world points that give good coverage for camera pose estimation.
    Creates points in a bounded volume that would typically be in front of camera.

    Args:
        n_points: Number of points to generate

    Returns:
        points: (n_points, 3) array of 3D world points
    """
    points = []

    # Define the boundaries of our world points volume
    x_range = (-5, 5)  # meters
    y_range = (-5, 5)  # meters
    z_range = (5, 15)  # meters, keep points in front of typical camera positions

    # Generate points in a structured pattern for better coverage
    n_per_dim = int(np.ceil(np.cbrt(n_points)))  # points per dimension to get roughly n_points total

    x_points = np.linspace(x_range[0], x_range[1], n_per_dim)
    y_points = np.linspace(y_range[0], y_range[1], n_per_dim)
    z_points = np.linspace(z_range[0], z_range[1], n_per_dim)

    # Create a grid of points
    for x in x_points:
        for y in y_points:
            for z in z_points:
                points.append([x, y, z])
                if len(points) >= n_points:
                    break
            if len(points) >= n_points:
                break
        if len(points) >= n_points:
            break

    points = np.array(points[:n_points])

    # Add some random perturbation to avoid perfectly regular structure
    perturbation = np.random.normal(0, 0.1, points.shape)
    points += perturbation

    # Add some additional random points if we didn't get enough from the grid
    while len(points) < n_points:
        new_point = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            np.random.uniform(*z_range)
        ])
        points = np.vstack([points, new_point])

    # Verify we have the right number of points
    assert len(points) == n_points, f"Generated {len(points)} points, expected {n_points}"

    return points

def test_point_generation():
    """Unit test for the point generation function"""
    n_points = 20
    points = generate_world_points(n_points)

    # Test shape
    assert points.shape == (n_points, 3), f"Wrong shape: {points.shape}"

    # Test bounds
    assert np.all(points[:, 0] >= -6) and np.all(points[:, 0] <= 6), "X bounds violated"
    assert np.all(points[:, 1] >= -6) and np.all(points[:, 1] <= 6), "Y bounds violated"
    assert np.all(points[:, 2] >= 4) and np.all(points[:, 2] <= 16), "Z bounds violated"

    # Test distribution
    x_spread = np.std(points[:, 0])
    y_spread = np.std(points[:, 1])
    z_spread = np.std(points[:, 2])

    assert x_spread > 1.0, "X spread too small"
    assert y_spread > 1.0, "Y spread too small"
    assert z_spread > 1.0, "Z spread too small"

    return points


def project_points(points_3d: np.ndarray, focal_length: float,
                  rotation: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D using perspective projection.
    OpenCV convention: +X right, +Y down, +Z forward
    """
    # Print debug info about inputs
    print("\nProject Points Debug:")
    print(f"Points 3D shape: {points_3d.shape}")
    print(f"First few 3D points:\n{points_3d[:3]}")
    print(f"Translation: {translation}")
    print(f"Rotation:\n{rotation}")

    # Reshape translation to (3,1) if it's (3,)
    if translation.shape == (3,):
        translation = translation.reshape(3, 1)

    # Transform points to camera frame
    points_cam = rotation @ points_3d.T + translation

    # Print intermediate results
    print(f"First few camera frame points:\n{points_cam[:, :3].T}")

    # Check which points are in front of camera and not too far
    valid_mask = (points_cam[2, :] > 0.1) & (points_cam[2, :] < 1000)

    # Project only valid points
    points_2d = np.zeros((len(points_3d), 2))
    if np.any(valid_mask):
        # Normalize by Z coordinate first
        points_normalized = points_cam[:2, valid_mask] / points_cam[2:3, valid_mask]
        # Then multiply by focal length
        points_2d[valid_mask] = focal_length * points_normalized.T

        # Print first few projections
        print(f"First few 2D projections:\n{points_2d[valid_mask][:3]}")

    return points_2d, valid_mask

def plot_camera_frame(ax, position, rotation, scale=0.5, label=None, color='r'):
    """
    Plot a camera coordinate frame using arrows for each axis.

    Args:
        ax: matplotlib 3D axis
        position: (3,) camera position
        rotation: (3,3) rotation matrix (camera to world)
        scale: length of coordinate axes
        label: label for legend
        color: base color for the frame
    """
    # Define axis colors
    colors = {'x': color, 'y': color, 'z': color}

    # Plot coordinate frame
    for axis, c in zip(range(3), ['x', 'y', 'z']):
        direction = rotation[:, axis] * scale
        ax.quiver(position[0], position[1], position[2],
                 direction[0], direction[1], direction[2],
                 color=colors[c], alpha=0.6,
                 label=label if axis == 0 else None)

def solve_pnp(points_3d, points_2d, focal_length):
    """Simple PnP solution using OpenCV"""
    # Full camera matrix with principal point at (0,0)
    camera_matrix = np.array([
        [focal_length, 0, 0],
        [0, focal_length, 0],
        [0, 0, 1]
    ])
    dist_coeffs = np.zeros(4)  # Assume no distortion

    # Print some debug info
    print(f"PnP input - 3D points shape: {points_3d.shape}")
    print(f"PnP input - 2D points shape: {points_2d.shape}")
    print(f"PnP input - First few 3D points:\n{points_3d[:3]}")
    print(f"PnP input - First few 2D points:\n{points_2d[:3]}")
    print(f"PnP input - focal length: {focal_length}")

    success, rvec, tvec = cv2.solvePnP(
        points_3d.astype(np.float32),
        points_2d.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if success:
        R = cv2.Rodrigues(rvec)[0]
        # Verify the solution makes sense
        reproj_points, _ = project_points(
            points_3d,
            focal_length,
            R,
            tvec.reshape(3)
        )
        reproj_error = np.mean(np.linalg.norm(reproj_points - points_2d, axis=1))
        print(f"PnP reprojection error: {reproj_error:.3f} pixels")

        # Print debug info about solution
        print(f"PnP solution - translation: {tvec.reshape(3)}")
        print(f"PnP solution - rotation matrix:\n{R}")
        return R, tvec.reshape(3)
    return None, None

class TestEKFPnP(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.focal_length = 800
        self.world_points = generate_world_points(20)
        self.ekf = EKFPnP(self.focal_length, self.world_points)

        # Generate trajectory
        self.positions, self.rotations = create_trajectory(100)

    def test_points_behind_camera(self):
        """Test handling of points that go behind the camera"""
        # Create a pose where some points will be behind camera
        position = np.array([0.0, 0.0, 0.0])  # Camera at origin
        # Rotation that will put some points behind camera
        rotation = Rotation.from_euler('y', 180, degrees=True).as_matrix()

        # Project points
        image_points, valid_mask = project_points(
            self.world_points,
            self.focal_length,
            rotation,
            position
        )

        # Verify some points are behind camera
        self.assertTrue(np.sum(~valid_mask) > 0,
                       "Test setup should have points behind camera")

        # Only use valid points for estimation
        valid_points = image_points[valid_mask]
        valid_world_points = self.world_points[valid_mask]

        # Create EKF with only valid points
        ekf = EKFPnP(self.focal_length, valid_world_points)

        # Initialize state
        initial_state = EKFState(
            s=np.array([0.1, 0.1, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            P=np.eye(13) * 0.1
        )

        # Add some noise to valid measurements
        noisy_points = valid_points + np.random.normal(0, 1.0, valid_points.shape)

        # Update should work with subset of valid points
        try:
            updated_state = ekf.update(initial_state, noisy_points)
        except Exception as e:
            self.fail(f"Update failed with valid points: {str(e)}")

    def test_moving_camera_tracking(self):
        """Test tracking a moving camera with point validity checks"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        dt = 0.1
        measurement_noise_std = 0.5

        # Initialize state at first position with better initial uncertainty
        initial_rotation = Rotation.from_matrix(self.rotations[0])
        initial_quat = initial_rotation.as_quat()

        initial_state = EKFState(
            s=np.array([
                *self.positions[0],  # Use true initial position
                initial_quat[3], *initial_quat[:3],  # Use true initial rotation
                0, 0, 0,  # Zero velocity
                0, 0, 0   # Zero angular velocity
            ]),
            P=np.diag([
                0.01, 0.01, 0.01,  # Smaller position uncertainty
                0.001, 0.001, 0.001, 0.001,  # Smaller rotation uncertainty
                0.1, 0.1, 0.1,  # Velocity uncertainty
                0.01, 0.01, 0.01  # Angular velocity uncertainty
            ])
        )

        state = initial_state
        position_errors = []
        rotation_errors = []
        n_valid_points = []

        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(121, projection='3d')  # 3D plot for camera pose
        ax2 = fig.add_subplot(122)  # 2D plot for projected points

        # Add text for instructions
        fig.text(0.5, 0.02, 'Press any key to advance to next frame (q to quit)',
                 ha='center', va='center')

        for i in range(1, len(self.positions)):
            # Clear previous plots
            ax1.clear()
            ax2.clear()

            # Plot 3D scene
            ax1.set_title('Camera Pose and World Points')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')

            # Plot world points
            ax1.scatter(self.world_points[:, 0],
                       self.world_points[:, 1],
                       self.world_points[:, 2],
                       c='blue', marker='.', alpha=0.3, label='World Points')

            # Plot true camera pose
            plot_camera_frame(ax1, self.positions[i], self.rotations[i],
                             scale=0.5, label='True', color='g')

            # Plot estimated camera pose
            estimated_rotation = Rotation.from_quat(
                [state.s[4], state.s[5], state.s[6], state.s[3]]
            ).as_matrix()
            plot_camera_frame(ax1, state.s[:3], estimated_rotation,
                             scale=0.5, label='Estimated', color='r')

            # Set reasonable axis limits
            ax1.set_xlim([-3, 3])
            ax1.set_ylim([-3, 3])
            ax1.set_zlim([0, 12])
            ax1.legend()

            # Generate measurements
            image_points, valid_mask = project_points(
                self.world_points,
                self.focal_length,
                self.rotations[i],
                self.positions[i]
            )

            # Plot 2D projections
            ax2.set_title('Image Plane Projections')
            ax2.set_xlabel('x (pixels)')
            ax2.set_ylabel('y (pixels)')

            if np.sum(valid_mask) >= 6:
                valid_points = image_points[valid_mask]
                valid_world_points = self.world_points[valid_mask]
                noisy_points = valid_points + np.random.normal(
                    0, measurement_noise_std, valid_points.shape
                )

                # Get PnP solution
                R_pnp, t_pnp = solve_pnp(valid_world_points, noisy_points, self.focal_length)

                # Get projections using estimated pose
                estimated_rotation = Rotation.from_quat(
                    [state.s[4], state.s[5], state.s[6], state.s[3]]
                ).as_matrix()
                estimated_position = state.s[:3]
                estimated_points, _ = project_points(
                    valid_world_points,
                    self.focal_length,
                    estimated_rotation,
                    estimated_position
                )

                if R_pnp is not None:
                    # Plot PnP solution
                    plot_camera_frame(ax1, t_pnp, R_pnp,
                                   scale=0.5, label='PnP', color='y')

                    # Project points using PnP solution
                    pnp_points, _ = project_points(
                        valid_world_points,
                        self.focal_length,
                        R_pnp,
                        t_pnp
                    )

                    # Plot PnP projections
                    ax2.scatter(pnp_points[:, 0], pnp_points[:, 1],
                              c='yellow', marker='.', label='PnP')

                    # Calculate and print PnP errors
                    pnp_errors = np.linalg.norm(pnp_points - valid_points, axis=1)
                    print(f"\nPnP average projection error: {np.mean(pnp_errors):.2f} pixels")

                    # Position difference
                    pos_diff = np.linalg.norm(t_pnp - self.positions[i])
                    print(f"PnP position error: {pos_diff:.3f} meters")
                    print(f"Ground truth position: {self.positions[i]}")
                    print(f"PnP position: {t_pnp}")

                    # Rotation difference (in degrees)
                    R_diff = R_pnp @ self.rotations[i].T
                    angle_diff = np.arccos((np.trace(R_diff) - 1) / 2)
                    print(f"PnP rotation error: {np.degrees(angle_diff):.2f} degrees")

                # Plot ground truth projections
                ax2.scatter(valid_points[:, 0], valid_points[:, 1],
                           c='green', marker='.', label='Ground Truth')

                # Plot estimated projections
                ax2.scatter(estimated_points[:, 0], estimated_points[:, 1],
                           c='blue', marker='.', label='Estimated')

                # Plot noisy measurements
                ax2.scatter(noisy_points[:, 0], noisy_points[:, 1],
                           c='red', marker='.', label='Measured')

                # Draw connecting lines and calculate distances
                total_distance = 0
                for i, (true_pt, est_pt, meas_pt) in enumerate(zip(valid_points, estimated_points, noisy_points)):
                    # Line from ground truth to measurement
                    ax2.plot([true_pt[0], meas_pt[0]], [true_pt[1], meas_pt[1]],
                             'r-', alpha=0.3, linewidth=0.5)

                    # Line from ground truth to estimate
                    ax2.plot([true_pt[0], est_pt[0]], [true_pt[1], est_pt[1]],
                             'b-', alpha=0.5, linewidth=1.0)

                    # Calculate and print distances
                    meas_dist = np.linalg.norm(true_pt - meas_pt)
                    est_dist = np.linalg.norm(true_pt - est_pt)
                    print(f"Point {i}: Measurement error = {meas_dist:.2f} pixels, Estimation error = {est_dist:.2f} pixels")
                    total_distance += est_dist

                avg_distance = total_distance / len(valid_points)
                print(f"Average estimation error: {avg_distance:.2f} pixels")

                # Set reasonable axis limits for image plane
                ax2.set_xlim(-self.focal_length, self.focal_length)
                ax2.set_ylim(-self.focal_length, self.focal_length)
                ax2.grid(True)
                ax2.legend()

                # Update EKF if measurements are valid
                if not np.any(np.abs(noisy_points) > self.focal_length * 10):
                    valid_world_points = self.world_points[valid_mask]
                    ekf_valid = EKFPnP(self.focal_length, valid_world_points)
                    R = np.eye(2 * np.sum(valid_mask)) * measurement_noise_std**2

                    try:
                        state = ekf_valid.update(state, noisy_points, R)

                        # Record metrics
                        position_error = np.linalg.norm(state.s[:3] - self.positions[i])
                        position_errors.append(position_error)
                        print(f"Current position error: {position_error}")
                        print(f"Number of valid points: {np.sum(valid_mask)}")

                        rotation_error = np.arccos(
                            (np.trace(estimated_rotation @ self.rotations[i].T) - 1) / 2
                        )
                        rotation_errors.append(rotation_error)
                        n_valid_points.append(np.sum(valid_mask))

                    except Exception as e:
                        print(f"Error during update at frame {i}: {str(e)}")
                        break

            plt.draw()
            plt.pause(0.001)  # Needed for updating the plot

            # Wait for keypress
            key = input("Press Enter for next frame (q to quit): ")
            if key.lower() == 'q':
                break

        plt.close()

        # Compute and check metrics
        if position_errors:
            avg_position_error = np.mean(position_errors)
            avg_rotation_error = np.mean(rotation_errors)
            avg_valid_points = np.mean(n_valid_points)

            print(f"\nFinal Statistics:")
            print(f"Average position error: {avg_position_error}")
            print(f"Average rotation error: {avg_rotation_error}")
            print(f"Average number of valid points: {avg_valid_points}")

            self.assertLess(avg_position_error, 0.5)
            self.assertLess(avg_rotation_error, 0.1)
            self.assertGreater(avg_valid_points, 6)
        else:
            self.fail("No valid tracking data collected")

if __name__ == '__main__':
    test = TestEKFPnP()
    test.setUp()
    test.test_moving_camera_tracking()
