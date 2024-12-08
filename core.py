import numpy as np
import sympy as sp
from scipy.integrate import odeint
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from scipy.spatial.transform import Rotation
from OpenGL.GL import shaders
import time

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

# Measurement model (measuring position and orientation)
h = sp.Matrix([x, y, z, qw, qx, qy, qz])

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
t = np.arange(0, 10, dt)  # simulation time

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
        self.true_color = (0.2, 0.8, 0.2)  # Green for true trajectory
        self.estimated_color = (0.8, 0.2, 0.2)  # Red for estimated trajectory
        self.measurement_color = (0.8, 0.8, 0.2)  # Yellow for measurements

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
                fragColor = vec4(0.2, 0.2, 0.2, 1.0);  // Runway color, fully opaque
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
        size = size / 2
        vertices = [
            ( size,  size,  size), ( size,  size, -size),
            ( size, -size, -size), ( size, -size,  size),
            (-size,  size,  size), (-size,  size, -size),
            (-size, -size, -size), (-size, -size,  size)
        ]

        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]

        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

    def draw_trajectory(self, points, color):
        """Draw trajectory line efficiently"""
        if len(points) < 2:
            return

        # Convert points to numpy array if not already
        points = np.asarray(points, dtype=np.float32)

        # Create and bind VAO if not exists
        if not hasattr(self, 'trajectory_vao'):
            self.trajectory_vao = glGenVertexArrays(1)
            self.trajectory_vbo = glGenBuffers(1)

        glBindVertexArray(self.trajectory_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.trajectory_vbo)
        glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STREAM_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glDisable(GL_LIGHTING)
        glColor3fv(color)
        glDrawArrays(GL_LINE_STRIP, 0, len(points))
        glEnable(GL_LIGHTING)

        glBindVertexArray(0)

    def draw_measurements(self, points, size=10):
        """Draw measurement points as coordinate frames"""
        glColor3f(*self.measurement_color)

        for point in points:
            glPushMatrix()
            glTranslatef(*point)

            # Draw coordinate frame
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

            glPopMatrix()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Check for escape key
                    return False
                elif event.key == pygame.K_r:  # Toggle runway
                    self.inset_show_runway = not self.inset_show_runway
                elif event.key == pygame.K_g:  # Toggle grid
                    self.inset_show_grid = not self.inset_show_grid
                elif event.key == pygame.K_x:  # Toggle axes
                    self.inset_show_axes = not self.inset_show_axes

            elif event.type == pygame.MOUSEBUTTONDOWN:
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
        # Save current viewport and matrices
        glPushAttrib(GL_VIEWPORT_BIT)
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
        glClear(GL_DEPTH_BUFFER_BIT)

        # Draw solid background first
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Disable lighting for background
        glDisable(GL_LIGHTING)

        # Draw dark grey background rectangle
        glColor3f(0.15, 0.15, 0.15)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()

        # Re-enable lighting for 3D scene
        glEnable(GL_LIGHTING)
        glClear(GL_DEPTH_BUFFER_BIT)

        # Set up camera projection for the scene
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.inset_size[0]/self.inset_size[1], 0.1, 2000.0)

        # Set up camera position/orientation
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Look along the trajectory
        gluLookAt(aircraft_position[0], aircraft_position[1], aircraft_position[2],
                  aircraft_position[0] + 10, aircraft_position[1], aircraft_position[2],
                  0, 1, 0)

        # Draw the scene elements
        # Always draw runway and axes
        self.draw_runway()
        self.draw_axes()
        # Only draw grid if enabled for insets
        if self.inset_show_grid:
            self.draw_grid()

        # Restore previous state
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

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

    def run_simulation(self, true_trajectory, estimated_trajectory, measurements=None):
        current_point = 0
        running = True
        clock = pygame.time.Clock()

        # Camera parameters
        camera_distance = 500
        azimuth = 45
        elevation = 30
        mouse_pressed = False
        last_mouse_pos = None
        zoom = 1.0

        # Initialize EKF state with initial bias
        state = np.zeros(10)
        state[:3] = true_trajectory[0] + np.random.normal(0, 5.0, 3)  # Add initial position error

        # Estimate initial velocity with bias
        velocity_window = 5
        initial_velocity = (true_trajectory[velocity_window] - true_trajectory[0]) / (velocity_window * dt)
        velocity_bias = np.random.normal(0, 2.0, 3)  # Add persistent velocity bias
        state[3:6] = initial_velocity + velocity_bias

        # For acceleration calculation
        prev_velocity = initial_velocity + velocity_bias

        initial_orientation = Rotation.from_euler('xyz', [0, 0, 0]).as_quat()
        state[6:10] = [initial_orientation[3], *initial_orientation[:3]]

        # Increase initial uncertainty
        P = np.eye(10) * 1.0  # Increased initial uncertainty
        Q = np.eye(10) * 0.5  # Increased process noise
        R = np.eye(7) * 0.1

        # Add persistent acceleration bias
        accel_bias = np.random.normal(0, 0.5, 3)

        # Pre-allocate arrays
        max_display_points = 100
        current_true = np.zeros((max_display_points, 3))
        current_estimated = np.zeros((max_display_points, 3))

        while running:
            loop_start = time.time()

            # Handle events
            event_start = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.inset_show_runway = not self.inset_show_runway
                    elif event.key == pygame.K_g:
                        self.inset_show_grid = not self.inset_show_grid
                    elif event.key == pygame.K_x:
                        self.inset_show_axes = not self.inset_show_axes
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_pressed = True
                        last_mouse_pos = pygame.mouse.get_pos()
                    elif event.button == 4:  # Mouse wheel up
                        zoom *= 0.9
                    elif event.button == 5:  # Mouse wheel down
                        zoom *= 1.1
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click release
                        mouse_pressed = False
                elif event.type == pygame.MOUSEMOTION and mouse_pressed:
                    current_mouse_pos = pygame.mouse.get_pos()
                    if last_mouse_pos is not None:
                        dx = current_mouse_pos[0] - last_mouse_pos[0]
                        dy = current_mouse_pos[1] - last_mouse_pos[1]
                        azimuth += dx * 0.5
                        elevation = max(-85, min(85, elevation + dy * 0.5))  # Clamp elevation
                    last_mouse_pos = current_mouse_pos
            event_time = time.time() - event_start

            # Calculate current velocity and acceleration with more noise
            if current_point < len(true_trajectory) - velocity_window:
                # Current velocity with more noise
                current_velocity = (true_trajectory[current_point + velocity_window] -
                                  true_trajectory[current_point]) / (velocity_window * dt)
                current_velocity += velocity_bias  # Add persistent bias

                # Acceleration from velocity difference
                acceleration = (current_velocity - prev_velocity) / dt
                acceleration_noise = np.random.normal(0, 1.0, 3)  # Increased noise
                acceleration += acceleration_noise + accel_bias  # Add noise and persistent bias

                # Update state velocity using acceleration
                state[3:6] += acceleration * dt

                # Add random walks to biases
                velocity_bias += np.random.normal(0, 0.01, 3)  # Slowly varying bias
                accel_bias += np.random.normal(0, 0.005, 3)

                # Store velocity for next iteration
                prev_velocity = current_velocity

            # EKF prediction with updated velocity
            state, P = ekf_predict(state, P, Q, dt)
            estimated_trajectory[current_point] = state[:3]

            # Calculate camera position
            camera_x = camera_distance * zoom * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
            camera_y = camera_distance * zoom * np.sin(np.radians(elevation))
            camera_z = camera_distance * zoom * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))

            # Clear and setup camera
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # Set camera position using spherical coordinates
            gluLookAt(camera_x, camera_y, camera_z,  # Camera position
                      0, 0, 0,                       # Look at center
                      0, 1, 0)                       # Up vector always points up

            # Draw scene elements
            self.draw_grid()
            self.draw_axes()
            self.draw_runway()

            # Update trajectories
            start_idx = max(0, current_point - max_display_points + 1)
            points_to_show = min(max_display_points, current_point + 1)

            if points_to_show > 0:
                current_true[:points_to_show] = true_trajectory[max(0, current_point - points_to_show + 1):current_point + 1]
                current_estimated[:points_to_show] = estimated_trajectory[max(0, current_point - points_to_show + 1):current_point + 1]

                self.draw_trajectory(current_true[:points_to_show], self.true_color)
                self.draw_robot(current_true[points_to_show-1], scale=20)

                if points_to_show > 1:
                    direction = current_true[points_to_show-1] - current_true[points_to_show-2]
                    aircraft_orientation = np.arctan2(direction[2], direction[0])
                else:
                    aircraft_orientation = 0
                self.draw_aircraft_view(current_true[points_to_show-1], aircraft_orientation, is_true=True)

                self.draw_trajectory(current_estimated[:points_to_show], self.estimated_color)
                self.draw_robot(current_estimated[points_to_show-1], scale=15)

                if points_to_show > 1:
                    direction = current_estimated[points_to_show-1] - current_estimated[points_to_show-2]
                    ekf_orientation = np.arctan2(direction[2], direction[0])
                else:
                    ekf_orientation = 0
                self.draw_aircraft_view(current_estimated[points_to_show-1], ekf_orientation, is_true=False)

            # Draw measurements
            current_measurement_idx = np.searchsorted(measurement_indices, current_point)
            if current_measurement_idx > 0:
                measurement_points = true_trajectory[measurement_indices[:current_measurement_idx]]
                self.draw_measurements(measurement_points)

            pygame.display.flip()
            current_point = (current_point + 1) % len(true_trajectory)
            clock.tick(15)

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

# Generate landing approach trajectory
t = np.arange(0, 20, 0.1)
initial_altitude = 100  # Start at 100m above runway
glide_slope = 3        # Standard 3-degree glide slope
initial_x = -(initial_altitude / np.tan(np.radians(glide_slope)))  # About -1908m for proper glide slope

# Create a glide path from (-1908, 100, 0) to (0, 0, 0)
x = initial_x * (1 - t/t[-1])  # Move from initial_x to 0
y = initial_altitude * (1 - t/t[-1])  # Descend from 100 to 0
z = np.zeros_like(t)   # Stay at z=0 (on centerline)

true_trajectory = np.column_stack((x, y, z))

# Generate noisy measurements every 20 steps
measurement_indices = np.arange(0, len(true_trajectory), 20)
measurement_noise = 4.0  # Reduced from 10.0
noisy_measurements = true_trajectory[measurement_indices] + np.random.normal(0, measurement_noise, (len(measurement_indices), 3))

# Calculate velocity estimates from position deltas
velocity_estimates = np.zeros_like(true_trajectory)
velocity_estimates[1:] = (true_trajectory[1:] - true_trajectory[:-1]) / dt
velocity_estimates[0] = velocity_estimates[1]

# Create non-Gaussian noise with multiple components
velocity_noise_gaussian = 2.0   # Reduced from 4.0
velocity_noise_uniform = 1.0    # Reduced from 2.0
velocity_noise_random_walk = 0.05  # Reduced from 0.1
velocity_bias = np.array([0.2, -0.1, 0.1])  # Reduced from [0.5, -0.3, 0.2]

# Initialize random walk
random_walk = np.zeros_like(velocity_estimates)
for i in range(1, len(random_walk)):
    random_walk[i] = random_walk[i-1] + np.random.normal(0, velocity_noise_random_walk, 3)

# Combine different noise sources
noisy_velocities = (
    velocity_estimates +
    velocity_bias +
    np.random.normal(0, velocity_noise_gaussian, velocity_estimates.shape) +
    np.random.uniform(-velocity_noise_uniform, velocity_noise_uniform, velocity_estimates.shape) +
    random_walk +
    np.random.exponential(2.0, velocity_estimates.shape)  # Add some exponential noise
)

def ekf_predict(state, P, Q, dt):
    """Prediction step of EKF using acceleration-based motion model"""
    # Unpack state
    x, y, z, vx, vy, vz, qw, qx, qy, qz = state

    # Calculate Jacobian F at current state
    F = F_num(x, y, z, vx, vy, vz, qw, qx, qy, qz)
    F = np.array(F, dtype=float)

    # State prediction
    state_pred = state.copy()
    # Update position with velocity
    state_pred[0:3] += state_pred[3:6] * dt + 0.5 * (state_pred[3:6] - state[3:6]) * dt**2
    # Velocity already updated in main loop using acceleration

    # Normalize quaternion
    quat_norm = np.linalg.norm(state_pred[6:10])
    if quat_norm > 0:
        state_pred[6:10] /= quat_norm

    # Covariance prediction
    P_pred = F @ P @ F.T + Q

    return state_pred, P_pred

def ekf_update(state, P, measurement, Q, R, dt):
    """Complete EKF step with prediction and update"""
    # First do prediction
    state_pred, P_pred = ekf_predict(state, P, Q, dt)

    # TEMPORARILY DISABLED: Measurement update
    """
    # Unpack predicted state
    x, y, z, vx, vy, vz, qw, qx, qy, qz = state_pred

    # Measurement update
    H = H_num(x, y, z, vx, vy, vz, qw, qx, qy, qz)
    H = np.array(H, dtype=float)

    # Innovation
    z_pred = H @ state_pred
    innovation = measurement - z_pred

    # Innovation covariance
    S = H @ P_pred @ H.T + R

    # Kalman gain
    K = P_pred @ H.T @ np.linalg.inv(S)

    # Update state and covariance
    state = state_pred + K @ innovation
    P = (np.eye(10) - K @ H) @ P_pred
    """

    # Just use prediction for now
    state = state_pred
    P = P_pred

    # Normalize quaternion after update
    quat_norm = np.linalg.norm(state[6:10])
    if quat_norm > 0:
        state[6:10] /= quat_norm

    return state, P

# First, update the measurement generation
def generate_measurements(true_trajectory, dt):
    # Generate measurements at 1/4 of the trajectory points
    measurement_indices = np.arange(0, len(true_trajectory), 4)

    # Calculate velocities from position deltas (only at measurement points)
    velocity_estimates = np.zeros((len(measurement_indices), 3))
    for i, idx in enumerate(measurement_indices[1:], 1):
        velocity_estimates[i] = (true_trajectory[idx] - true_trajectory[idx-4]) / (4 * dt)
    velocity_estimates[0] = velocity_estimates[1]  # Copy first velocity

    # Generate noise components
    velocity_noise_gaussian = 2.0
    velocity_noise_uniform = 1.0
    velocity_noise_random_walk = 0.05
    velocity_bias = np.array([0.2, -0.1, 0.1])

    # Initialize random walk
    random_walk = np.zeros_like(velocity_estimates)
    for i in range(1, len(random_walk)):
        random_walk[i] = random_walk[i-1] + np.random.normal(0, velocity_noise_random_walk, 3)

    # Combine different noise sources for velocity
    noisy_velocities = (
        velocity_estimates +
        velocity_bias +
        np.random.normal(0, velocity_noise_gaussian, velocity_estimates.shape) +
        np.random.uniform(-velocity_noise_uniform, velocity_noise_uniform, velocity_estimates.shape) +
        random_walk
    )

    # Generate orientation measurements
    orientation_measurements = np.zeros((len(measurement_indices), 4))
    orientation_noise = 0.05
    initial_orientation = Rotation.from_euler('xyz', [0, 0, 0]).as_quat()
    base_orientation = np.array([initial_orientation[3], *initial_orientation[:3]])

    for i in range(len(measurement_indices)):
        euler_noise = np.random.normal(0, orientation_noise, 3)
        noisy_orientation = Rotation.from_euler('xyz', euler_noise).as_quat()
        orientation_measurements[i] = [noisy_orientation[3], *noisy_orientation[:3]]

    # Combine velocity and orientation measurements
    measurements = np.zeros((len(measurement_indices), 7))
    measurements[:, :3] = noisy_velocities
    measurements[:, 3:] = orientation_measurements

    return measurements, measurement_indices

# Generate measurements
measurements, measurement_indices = generate_measurements(true_trajectory, dt)

# Initialize EKF state and run filter
estimated_trajectory = np.zeros_like(true_trajectory)
state = np.zeros(10)
state[:3] = true_trajectory[0]  # Initial position
initial_orientation = Rotation.from_euler('xyz', [0, 0, 0]).as_quat()
state[6:10] = [initial_orientation[3], *initial_orientation[:3]]  # [qw, qx, qy, qz]

P = np.eye(10) * 0.5  # Initial state covariance
Q = np.eye(10) * 0.05  # Process noise covariance
R = np.eye(7) * measurement_noise  # Measurement noise covariance

# Run EKF for each timestep
measurement_idx = 0
for i in range(len(true_trajectory)):
    # Only update with measurement when we have one
    if measurement_idx < len(measurement_indices) and i == measurement_indices[measurement_idx]:
        state, P = ekf_update(state, P, measurements[measurement_idx], Q, R, dt)
        measurement_idx += 1
    else:
        # Prediction only when no measurement
        state, P = ekf_predict(state, P, Q, dt)

    estimated_trajectory[i] = state[:3]

# Create and run visualizer with both trajectories
visualizer = RobotVisualizer()
visualizer.run_simulation(true_trajectory, estimated_trajectory, noisy_measurements)
