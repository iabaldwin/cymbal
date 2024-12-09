import time
import math

from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
from pygame.locals import *
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
import numpy as np
import pygame
import sympy as sp

# Define symbolic variables for state and measurement
x, y, z = sp.symbols('x y z')  # positions
vx, vy, vz = sp.symbols('vx vy vz')  # velocities
qw, qx, qy, qz = sp.symbols('qw qx qy qz')  # quaternion orientation

# State vector X = [x, y, z, vx, vy, vz, qw, qx, qy, qz]
state = sp.Matrix([x, y, z, vx, vy, vz, qw, qx, qy, qz])

# Process model (constant velocity): dx/dt = vx, dy/dt = vy, dz/dt = dz
# d(vx)/dt = 0, d(vy)/dt = 0, d(vz)/dt = 0
f = sp.Matrix([
    vx,
    vy,
    vz,
    0,
    0,
    0,
    0,  # quaternion derivatives are zero for now (constant orientation)
    0,
    0,
    0
])

# Measurement model (measuring position only)
h = sp.Matrix([x, y, z])

# Calculate Jacobians
F = f.jacobian(state)
H = h.jacobian(state)

print("Process model Jacobian F:")
print(F)
print("\nMeasurement model Jacobian H:")
print(H)

# Convert symbolic expressions to numerical functions
F_num = sp.lambdify((x, y, z, vx, vy, vz, qw, qx, qy, qz), F, 'numpy')
H_num = sp.lambdify((x, y, z, vx, vy, vz, qw, qx, qy, qz), H, 'numpy')

# Simulation parameters
dt = 0.1  # time step
t = np.arange(0, 60, dt)  # Increased simulation time to 60 seconds (from 20)

def ekf_predict(state, P, Q, dt):
    """Prediction step of EKF using constant velocity model"""
    # Unpack state
    x, y, z, vx, vy, vz, qw, qx, qy, qz = state

    print(f"\nBefore prediction:")
    print(f"Position: ({x:.2f}, {y:.2f}, {z:.2f})")
    print(f"Velocity: ({vx:.2f}, {vy:.2f}, {vz:.2f})")

    # State prediction using process model f
    # dx/dt = vx, dy/dt = vy, dz/dt = vz
    # d(vx)/dt = 0, d(vy)/dt = 0, d(vz)/dt = 0
    # d(qw)/dt = 0, d(qx)/dt = 0, d(qy)/dt = 0, d(qz)/dt = 0
    state_pred = np.zeros_like(state)

    # Update positions using current velocities
    state_pred[0] = x + vx * dt  # x += vx*dt
    state_pred[1] = y + vy * dt  # y += vy*dt
    state_pred[2] = z + vz * dt  # z += vz*dt

    # Keep velocities constant
    state_pred[3:6] = state[3:6]

    # Keep quaternion constant
    state_pred[6:10] = state[6:10]

    # Normalize quaternion
    quat_norm = np.linalg.norm(state_pred[6:10])
    if quat_norm > 0:
        state_pred[6:10] /= quat_norm

    print(f"\nAfter prediction:")
    print(f"Position: ({state_pred[0]:.2f}, {state_pred[1]:.2f}, {state_pred[2]:.2f})")
    print(f"Velocity: ({state_pred[3]:.2f}, {state_pred[4]:.2f}, {state_pred[5]:.2f})")
    print(f"dt: {dt}")

    # Calculate Jacobian F at current state
    F = np.array(F_num(x, y, z, vx, vy, vz, qw, qx, qy, qz), dtype=float)

    # Covariance prediction
    P_pred = F @ P @ F.T + Q

    return state_pred, P_pred

def ekf_update(state_pred, P_pred, measurement, R):
    """Update step of EKF"""
    # Unpack predicted state
    x, y, z, vx, vy, vz, qw, qx, qy, qz = state_pred

    # Calculate measurement Jacobian H at current state
    H = np.array(H_num(x, y, z, vx, vy, vz, qw, qx, qy, qz), dtype=float)

    # Innovation (measurement residual)
    z_pred = state_pred[:3]  # Only predict position
    innovation = measurement - z_pred

    # Innovation covariance
    S = H @ P_pred @ H.T + R

    # Kalman gain
    K = P_pred @ H.T @ np.linalg.inv(S)

    # Update state and covariance
    state = state_pred + K @ innovation
    P = (np.eye(10) - K @ H) @ P_pred

    # Normalize quaternion after update
    quat_norm = np.linalg.norm(state[6:10])
    if quat_norm > 0:
        state[6:10] /= quat_norm

    return state, P

# True initial state
x0 = np.array([0, 0, 0, 1, 0.5, 0.25, 1, 0, 0, 0])  # starting at origin with some velocity

# Function for numerical integration
def robot_motion(state, t):
    return [state[3], state[4], state[5], 0, 0, 0, 0, 0, 0, 0]

# Generate true trajectory
true_states = odeint(robot_motion, x0, t)

# Add noise to velocity measurements (separate position and orientation noise)
position_noise = 0.1
orientation_noise = 0.05

# Generate noisy measurements for position and orientation
measurements = np.zeros((len(t), 7))
measurements[:, :3] = true_states[:, :3] + np.random.normal(0, position_noise, (len(t), 3))  # Position measurements

# Generate orientation measurements (assuming constant orientation for now)
initial_orientation = Rotation.from_euler('xyz', [0, 0, 0]).as_quat()  # [qx, qy, qz, qw]
base_orientation = np.array([initial_orientation[3], *initial_orientation[:3]])  # [qw, qx, qy, qz]
for i in range(len(t)):
    # Add some noise to orientation
    euler_noise = np.random.normal(0, orientation_noise, 3)
    noisy_orientation = Rotation.from_euler('xyz', euler_noise).as_quat()
    measurements[i, 3:] = [noisy_orientation[3], *noisy_orientation[:3]]  # [qw, qx, qy, qz]

class RobotVisualizer:
    def __init__(self, width=1600, height=1200):
        pygame.init()

        # Set up display with multisampling for antialiasing
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Robot Trajectory")

        # Store window dimensions
        self.width = width
        self.height = height

        # Mouse and camera control attributes
        self.prev_mouse_pos = None
        self.mouse_button_down = False
        self.camera_distance = 500
        self.rotation_x = 30
        self.rotation_y = 45
        self.look_at = [0, 0, 0]
        self.up_vector = [0, 1, 0]

        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, (1000.0, 1000.0, 1000.0, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.3, 0.3, 0.3, 1.0))
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.1, 0.1, 0.1, 1.0))

        # Initialize view settings
        self.inset_show_grid = True
        self.inset_show_runway = True
        self.inset_show_axes = True

        # Set up the scene
        self.resize(width, height)

        # Initialize runway parameters
        self.runway_length = 400
        self.runway_width = 30

        # Initialize terrain
        self.terrain_size = 1000
        self.terrain_step = 50
        self.terrain_grid = self.generate_terrain()

        # Initialize shader and buffers after OpenGL context is created
        self.init_shaders()

        # Set up inset view positions and size
        self.inset_size = (200, 150)
        self.true_inset_position = (10, 10)
        self.ekf_inset_position = (220, 10)

        # Colors
        self.true_color = (1, 0.5, 0)      # Orange for true trajectory
        self.estimated_color = (0, 1, 0.5)  # Cyan for estimated trajectory
        self.measurement_color = (1, 0, 1)  # Magenta for measurements
        self.measurement_size = 5.0

        # Add these lines to the existing __init__ method
        self.auto_orbit = True
        self.orbit_speed = 0.15  # degrees per frame
        self.orbit_time = 0

    def init_shaders(self):
        """Initialize shaders and store uniform locations"""
        self.grid_shader = self.create_grid_shader()
        self.setup_grid_buffers()

        # Store shader uniform locations
        self.proj_loc = glGetUniformLocation(self.grid_shader, "projection")
        self.mv_loc = glGetUniformLocation(self.grid_shader, "modelview")

    def resize(self, width, height):
        """Handle window resize"""
        if height == 0:
            height = 1

        # Update viewport
        glViewport(0, 0, width, height)

        # Set up perspective projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, float(width)/float(height), 0.1, 2000.0)

        # Reset modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Store new dimensions
        self.width = width
        self.height = height

    def create_grid_shader(self):
        vertex_shader = """
        #version 330
        layout(location = 0) in vec3 position;
        layout(location = 1) in float height;

        uniform mat4 projection;
        uniform mat4 modelview;

        out float v_height;

        void main() {
            gl_Position = projection * modelview * vec4(position, 1.0);
            v_height = height;
        }
        """

        fragment_shader = """
        #version 330
        in float v_height;

        out vec4 fragColor;

        void main() {
            if (v_height == 0.0) {
                // Make runway area fully transparent
                fragColor = vec4(0.0, 0.0, 0.0, 0.0);
            } else {
                float t = v_height / 25.0;  // Normalize by max height
                float alpha = min(0.2 + t * 0.8, 1.0);  // More opaque as height increases
                // Grey to blue interpolation with transparency
                fragColor = vec4(0.3*(1.0-t), 0.3*(1.0-t), 0.3 + 0.7*t, alpha);
            }
        }
        """

        # Compile shaders
        shader = shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )

        return shader

    def setup_grid_buffers(self):
        grid_points = self.terrain_size // self.terrain_step + 1
        half_grid = grid_points // 2

        # Generate grid points using numpy
        x = np.linspace(-half_grid * self.terrain_step, half_grid * self.terrain_step, grid_points)
        z = np.linspace(-half_grid * self.terrain_step, half_grid * self.terrain_step, grid_points)
        X, Z = np.meshgrid(x, z)

        # Create vertices for triangle strips
        vertices = []
        heights = []

        # Create triangle strips
        for i in range(grid_points-1):
            for j in range(grid_points):
                # Add two vertices for each column (current and next row)
                vertices.append([X[i,j], self.terrain_grid[i,j], Z[i,j]])
                vertices.append([X[i+1,j], self.terrain_grid[i+1,j], Z[i+1,j]])
                heights.append(self.terrain_grid[i,j])
                heights.append(self.terrain_grid[i+1,j])

        vertices = np.array(vertices, dtype=np.float32)
        heights = np.array(heights, dtype=np.float32)

        # Create and bind VAO
        self.grid_vao = glGenVertexArrays(1)
        glBindVertexArray(self.grid_vao)

        # Vertex buffer
        self.grid_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.grid_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Height buffer
        self.height_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.height_vbo)
        glBufferData(GL_ARRAY_BUFFER, heights.nbytes, heights, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # Store the number of vertices per strip and number of strips
        self.vertices_per_strip = grid_points * 2
        self.num_strips = grid_points - 1

        glBindVertexArray(0)

    def draw_grid(self):
        """Draw the terrain grid using triangle strips"""
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self.grid_shader)

        # Set uniforms using stored locations
        proj_matrix = glGetFloatv(GL_PROJECTION_MATRIX)
        mv_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, proj_matrix)
        glUniformMatrix4fv(self.mv_loc, 1, GL_FALSE, mv_matrix)

        # Draw grid using triangle strips
        glBindVertexArray(self.grid_vao)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # Draw wireframe

        for i in range(self.num_strips):
            glDrawArrays(GL_TRIANGLE_STRIP, i * self.vertices_per_strip, self.vertices_per_strip)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # Reset polygon mode
        glBindVertexArray(0)
        glUseProgram(0)

        # Disable blending
        glDisable(GL_BLEND)

    def draw_axes(self, size=100):
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(size, 0, 0)
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, size, 0)
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, size)
        glEnd()

    def draw_robot(self, position, scale=10):
        """Draw the robot as coordinate axes"""
        glPushMatrix()
        glTranslatef(*position)

        # Draw coordinate axes
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(scale, 0, 0)
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, scale, 0)
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, scale)
        glEnd()

        glPopMatrix()

    def draw_cube(self, size):
        """Draw a cube centered at the origin"""
        half = size / 2
        vertices = [
            [-half, -half, -half], [half, -half, -half],
            [half, half, -half], [-half, half, -half],
            [-half, -half, half], [half, -half, half],
            [half, half, half], [-half, half, half]
        ]

        # Define faces using vertex indices
        faces = [
            [0, 1, 2, 3],  # Front
            [1, 5, 6, 2],  # Right
            [5, 4, 7, 6],  # Back
            [4, 0, 3, 7],  # Left
            [3, 2, 6, 7],  # Top
            [4, 5, 1, 0]   # Bottom
        ]

        glBegin(GL_QUADS)
        for face in faces:
            for vertex in face:
                glVertex3f(*vertices[vertex])
        glEnd()

    def draw_trajectory(self, points, color):
        glColor3f(*color)
        glBegin(GL_LINE_STRIP)
        for point in points:
            glVertex3f(*point)
        glEnd()

    def draw_measurements(self, points):
        """Draw measurement points as small cubes"""
        glColor3f(*self.measurement_color)
        for point in points:
            glPushMatrix()
            glTranslatef(*point)
            self.draw_cube(self.measurement_size)
            glPopMatrix()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Escape key to exit
                    return False
                elif event.key == pygame.K_r:  # Toggle runway
                    self.inset_show_runway = not self.inset_show_runway
                elif event.key == pygame.K_g:  # Toggle grid
                    self.inset_show_grid = not self.inset_show_grid
                elif event.key == pygame.K_x:  # Toggle axes
                    self.inset_show_axes = not self.inset_show_axes
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Disable auto-orbit on any mouse click
                self.auto_orbit = False
                if event.button in [1, 2, 3]:
                    self.mouse_button_down = event.button
                    self.prev_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_button_down = False
                self.prev_mouse_pos = None
            elif event.type == pygame.MOUSEMOTION and self.prev_mouse_pos:
                dx = event.pos[0] - self.prev_mouse_pos[0]
                dy = event.pos[1] - self.prev_mouse_pos[1]

                if self.mouse_button_down == 1:  # Left click - rotate
                    self.rotation_y += dx * 0.5
                    self.rotation_x += dy * 0.5
                elif self.mouse_button_down == 3:  # Right click - pan
                    self.look_at[0] -= dx * 0.5
                    self.look_at[1] += dy * 0.5

                self.prev_mouse_pos = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                self.camera_distance -= event.y * 20
                self.camera_distance = max(10, min(1000, self.camera_distance))

        return True

    def draw_aircraft_view(self, aircraft_position, aircraft_orientation, is_true=True):
        """Draw the view from aircraft's camera"""
        # Save ALL current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)

        # Save matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        # Set up viewport for inset based on whether it's true or estimated view
        position = self.true_inset_position if is_true else self.ekf_inset_position
        glViewport(position[0],
                  self.height - position[1] - self.inset_size[1],
                  self.inset_size[0], self.inset_size[1])

        # Clear depth buffer for this viewport
        glScissor(position[0],
                 self.height - position[1] - self.inset_size[1],
                 self.inset_size[0], self.inset_size[1])
        glEnable(GL_SCISSOR_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)

        # Draw semi-transparent black background for inset
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.inset_size[0], 0, self.inset_size[1], -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Disable depth test and lighting for background quad
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glColor4f(0, 0, 0, 1.0)  # Fully opaque black
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.inset_size[0], 0)
        glVertex2f(self.inset_size[0], self.inset_size[1])
        glVertex2f(0, self.inset_size[1])
        glEnd()

        # Re-enable depth test and lighting for 3D scene
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        # Set up camera projection for the scene
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.inset_size[0]/self.inset_size[1], 0.1, 2000.0)

        # Set up camera position/orientation
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Look along the trajectory
        gluLookAt(aircraft_position[0], aircraft_position[1], aircraft_position[2],
                 aircraft_position[0] + 10*math.cos(aircraft_orientation),
                 aircraft_position[1],
                 aircraft_position[2] + 10*math.sin(aircraft_orientation),
                 0, 1, 0)

        # Draw the scene elements
        if self.inset_show_grid:
            self.draw_grid()
        if self.inset_show_axes:
            self.draw_axes()
        if self.inset_show_runway:
            self.draw_runway()

        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # Disable scissor test before restoring state
        glDisable(GL_SCISSOR_TEST)

        # Restore ALL OpenGL state
        glPopAttrib()

        # Reset viewport to full window
        glViewport(0, 0, self.width, self.height)

    def project_runway_to_camera(self, aircraft_position, aircraft_orientation):
        """Project runway corners to aircraft camera view"""
        # Define runway corners in world coordinates
        l, w = self.runway_length/2, self.runway_width/2
        corners = np.array([
            [-l, 0, -w],
            [l, 0, -w],
            [l, 0, w],
            [-l, 0, w]
        ])

        # Transform to camera coordinates
        camera_points = []
        for corner in corners:
            # Vector from aircraft to corner
            v = corner - aircraft_position

            # Simple perspective projection
            if v[2] != 0:  # Avoid division by zero
                # Scale factor based on distance
                scale = 1.0 / max(1, abs(v[2]))
                x = self.inset_size[0] * v[0] * scale
                y = self.inset_size[1] * v[1] * scale
                camera_points.append((
                    int(x + self.inset_size[0]/2),
                    int(y + self.inset_size[1]/2)
                ))

        return camera_points

    def draw_runway(self):
        """Draw runway as a rectangle on the ground"""
        # Draw main runway surface
        glColor3f(0.2, 0.2, 0.2)  # Darker gray for better contrast
        l, w = self.runway_length/2, self.runway_width/2

        glBegin(GL_QUADS)
        glVertex3f(-l, 0, -w)
        glVertex3f(l, 0, -w)
        glVertex3f(l, 0, w)
        glVertex3f(-l, 0, w)
        glEnd()

        # Draw runway edge lines
        glColor3f(1, 1, 1)  # White
        glBegin(GL_LINES)
        # Long edges
        glVertex3f(-l, 0, -w)
        glVertex3f(l, 0, -w)
        glVertex3f(-l, 0, w)
        glVertex3f(l, 0, w)
        # Threshold lines
        glVertex3f(-l, 0, -w)
        glVertex3f(-l, 0, w)
        glVertex3f(l, 0, -w)
        glVertex3f(l, 0, w)
        glEnd()

    def generate_terrain(self):
        grid_points = self.terrain_size // self.terrain_step + 1
        # Generate random heights with halved maximum height
        terrain = np.random.uniform(0, 25, (grid_points, grid_points))  # Halved from 50

        # Create a flat area for the runway and surrounding buffer
        center = grid_points // 2

        # Make flat area 3x the runway length in all directions
        buffer_length = self.runway_length * 3
        runway_width_cells = int(self.runway_width / self.terrain_step) + 2
        runway_length_cells = int(buffer_length / self.terrain_step) + 2

        # Make sure buffer doesn't exceed grid size
        max_runway_size = (grid_points - 4) // 2  # Leave room for transition
        runway_length_cells = min(runway_length_cells, max_runway_size)
        runway_width_cells = min(runway_width_cells, max_runway_size)

        # Set runway and buffer area to zero elevation
        half_width = runway_width_cells // 2
        half_length = runway_length_cells // 2

        # Ensure indices are within bounds - now aligned with runway direction
        start_x = max(0, center - half_width)  # Width now on X axis
        end_x = min(grid_points, center + half_width + 1)
        start_z = max(0, center - half_length)  # Length now on Z axis
        end_z = min(grid_points, center + half_length + 1)

        terrain[start_x:end_x, start_z:end_z] = 0

        # Create a gradual slope from the buffer to the terrain
        buffer_size = min(3, (grid_points - max(runway_width_cells, runway_length_cells)) // 2)

        for i in range(buffer_size):
            factor = (i + 1) / (buffer_size + 1)

            # Calculate safe indices for each edge
            left_col = max(0, start_x - i - 1)
            right_col = min(grid_points - 1, end_x + i)
            front_row = max(0, start_z - i - 1)
            back_row = min(grid_points - 1, end_z + i)

            # Apply gradual slope with bounds checking
            if left_col >= 0:
                terrain[left_col, start_z:end_z] *= factor
            if right_col < grid_points:
                terrain[right_col, start_z:end_z] *= factor
            if front_row >= 0:
                terrain[start_x:end_x, front_row] *= factor
            if back_row < grid_points:
                terrain[start_x:end_x, back_row] *= factor

        return terrain

    def draw_help_text(self, lines):
        """Draw help text in the corner of the screen"""
        font = pygame.font.Font(None, 24)
        y = 20
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            pygame.display.get_surface().blit(text, (10, y))
            y += 20

    def run_simulation(self, true_trajectory, estimated_trajectory, measurements=None, initial_state=None):
        current_point = 0
        running = True
        clock = pygame.time.Clock()

        # Initialize EKF state
        if initial_state is None:
            state = np.zeros(10)
            state[:3] = true_trajectory[0]  # Initial position
            state[3:6] = np.array([1.0, -0.5, 0.0])  # Initial velocity guess
            initial_orientation = Rotation.from_euler('xyz', [0, 0, 0]).as_quat()
            state[6:10] = [initial_orientation[3], *initial_orientation[:3]]  # [qw, qx, qy, qz]
        else:
            state = initial_state.copy()

        # Initialize covariances with higher uncertainty for unobserved states
        P = np.eye(10)
        P[:3, :3] *= 1.0    # Position uncertainty (observed)
        P[3:6, 3:6] *= 2.0  # Velocity uncertainty (unobserved but correlated with position)
        P[6:, 6:] *= 5.0    # Orientation uncertainty (unobserved)

        # Process noise - keep this relatively low to trust our motion model
        Q = np.eye(10)
        Q[:3, :3] *= 0.1     # Position process noise (low - trust the motion model)
        Q[3:6, 3:6] *= 0.5   # Velocity process noise (moderate - allow some velocity changes)
        Q[6:, 6:] *= 0.01    # Orientation process noise (very low - assume constant)

        # Measurement noise - make this high to trust measurements less
        measurement_noise = 10.0  # Noise standard deviation for position measurements
        R = np.eye(3) * (measurement_noise * 5.0)**2  # Much higher measurement uncertainty

        while running:
            loop_start = time.time()

            # Handle input events
            input_start = time.time()
            running = self.handle_input()
            input_time = time.time() - input_start

            # Clear screen and set camera
            gl_start = time.time()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Modify the camera position calculation
            if self.auto_orbit:
                # Update orbit time
                self.orbit_time += 1
                # Auto-rotate the camera
                self.rotation_y = self.orbit_time * self.orbit_speed

            # Calculate camera position
            cam_x = self.camera_distance * math.sin(math.radians(self.rotation_y)) * math.cos(math.radians(self.rotation_x))
            cam_y = self.camera_distance * math.sin(math.radians(self.rotation_x))
            cam_z = self.camera_distance * math.cos(math.radians(self.rotation_y)) * math.cos(math.radians(self.rotation_x))

            gluLookAt(cam_x + self.look_at[0], cam_y + self.look_at[1], cam_z + self.look_at[2],
                     *self.look_at,
                     *self.up_vector)
            gl_setup_time = time.time() - gl_start

            # Draw scene elements
            draw_start = time.time()
            self.draw_grid()
            self.draw_axes()
            self.draw_runway()
            draw_scene_time = time.time() - draw_start

            # EKF step
            ekf_start = time.time()

            # First predict
            state_pred, P_pred = ekf_predict(state, P, Q, dt)

            # Then update if measurement is available
            current_measurement_idx = np.searchsorted(measurement_indices, current_point)
            if current_measurement_idx < len(measurements) and measurement_indices[current_measurement_idx] == current_point:
                # Get current measurement (only position)
                meas = measurements[current_measurement_idx]
                state, P = ekf_update(state_pred, P_pred, meas, R)
            else:
                # No measurement, just use prediction
                state = state_pred
                P = P_pred

            ekf_time = time.time() - ekf_start

            # Store estimated position
            estimated_trajectory[current_point] = state[:3]

            # Draw trajectories
            traj_start = time.time()
            current_true = true_trajectory[:current_point+1]
            current_estimated = estimated_trajectory[:current_point+1]

            if len(current_true) > 0:
                self.draw_trajectory(current_true, self.true_color)
                self.draw_robot(current_true[-1], scale=20)

                # Calculate true orientation from trajectory
                if len(current_true) > 1:
                    direction = current_true[-1] - current_true[-2]
                    true_orientation = math.atan2(direction[2], direction[0])
                else:
                    true_orientation = 0

            if len(current_estimated) > 0:
                self.draw_trajectory(current_estimated, self.estimated_color)
                self.draw_robot(current_estimated[-1], scale=15)

                # Calculate estimated orientation from trajectory
                if len(current_estimated) > 1:
                    direction = current_estimated[-1] - current_estimated[-2]
                    estimated_orientation = math.atan2(direction[2], direction[0])
                else:
                    estimated_orientation = 0
            traj_time = time.time() - traj_start

            # Draw measurements
            meas_start = time.time()
            if current_measurement_idx > 0:
                measurement_points = measurements[:current_measurement_idx]
                self.draw_measurements(measurement_points)
            meas_time = time.time() - meas_start

            # Draw inset views
            if len(current_true) > 0:
                self.draw_aircraft_view(current_true[-1], true_orientation, is_true=True)
            if len(current_estimated) > 0:
                self.draw_aircraft_view(current_estimated[-1], estimated_orientation, is_true=False)

            # Flip display buffers
            flip_start = time.time()
            pygame.display.flip()
            flip_time = time.time() - flip_start

            # Calculate and print frame time
            frame_time = time.time() - loop_start
            print(f"Frame {current_point:4d} timing (ms): "
                  f"Total={frame_time*1000:.1f}, "
                  f"Input={input_time*1000:.1f}, "
                  f"GL Setup={gl_setup_time*1000:.1f}, "
                  f"Scene={draw_scene_time*1000:.1f}, "
                  f"EKF={ekf_time*1000:.1f}, "
                  f"Traj={traj_time*1000:.1f}, "
                  f"Meas={meas_time*1000:.1f}, "
                  f"Flip={flip_time*1000:.1f}")

            current_point = (current_point + 1) % len(true_trajectory)
            clock.tick(30)  # Increased to 30 FPS for smoother animation

        pygame.quit()

if __name__ == '__main__':
    # Generate landing approach trajectory
    t = np.arange(0, 60, dt)
    initial_altitude = 100  # Start at 100m above runway
    glide_slope = 3        # Standard 3-degree glide slope
    initial_x = -(initial_altitude / np.tan(np.radians(glide_slope)))  # About -1908m for proper glide slope

    # Create a glide path from (-1908, 100, 0) to (0, 0, 0)
    x = initial_x * (1 - t/t[-1])  # Move from initial_x to 0
    y = initial_altitude * (1 - t/t[-1])  # Descend from 100 to 0
    z = np.zeros_like(t)   # Stay at z=0 (on centerline)

    true_trajectory = np.column_stack((x, y, z))

    # Calculate initial velocities from trajectory
    initial_vx = -initial_x / t[-1]  # Velocity needed to reach x=0
    initial_vy = -initial_altitude / t[-1]  # Velocity needed to reach y=0
    initial_vz = 0  # No lateral movement

    print(f"Initial velocities: vx={initial_vx:.2f}, vy={initial_vy:.2f}, vz={initial_vz:.2f}")

    # Generate noisy measurements every 20 steps
    measurement_indices = np.arange(0, len(true_trajectory), 20)
    measurement_noise = 10.0  # Noise standard deviation for position measurements
    noisy_measurements = true_trajectory[measurement_indices] + np.random.normal(0, measurement_noise, (len(measurement_indices), 3))

    # Initialize EKF state
    initial_state = np.zeros(10)
    initial_state[:3] = true_trajectory[0]  # Initial position
    initial_state[3:6] = np.array([initial_vx, initial_vy, initial_vz])  # Set velocities to match trajectory
    initial_orientation = Rotation.from_euler('xyz', [0, 0, 0]).as_quat()
    initial_state[6:10] = [initial_orientation[3], *initial_orientation[:3]]  # [qw, qx, qy, qz]

    # Initialize estimated trajectory array
    estimated_trajectory = np.zeros_like(true_trajectory)

    # Create and run visualizer with both trajectories
    visualizer = RobotVisualizer()
    visualizer.run_simulation(true_trajectory, estimated_trajectory, noisy_measurements, initial_state)
