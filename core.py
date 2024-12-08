import numpy as np
import sympy as sp
from scipy.integrate import odeint
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Define symbolic variables for state and measurement
x, y, z = sp.symbols('x y z')  # positions
vx, vy, vz = sp.symbols('vx vy vz')  # velocities

# State vector X = [x, y, z, vx, vy, vz]
state = sp.Matrix([x, y, z, vx, vy, vz])

# Process model (constant velocity): dx/dt = vx, dy/dt = vy, dz/dt = dz
# d(vx)/dt = 0, d(vy)/dt = 0, d(vz)/dt = 0
f = sp.Matrix([
    vx,
    vy,
    vz,
    0,
    0,
    0
])

# Measurement model (measuring velocities)
h = sp.Matrix([vx, vy, vz])

# Calculate Jacobians
F = f.jacobian(state)
H = h.jacobian(state)

print("Process model Jacobian F:")
print(F)
print("\nMeasurement model Jacobian H:")
print(H)

# Convert symbolic expressions to numerical functions
F_num = sp.lambdify((x, y, z, vx, vy, vz), F, 'numpy')
H_num = sp.lambdify((x, y, z, vx, vy, vz), H, 'numpy')

# Simulation parameters
dt = 0.1  # time step
t = np.arange(0, 10, dt)  # simulation time

# True initial state
x0 = np.array([0, 0, 0, 1, 0.5, 0.25])  # starting at origin with some velocity

# Function for numerical integration
def robot_motion(state, t):
    return [state[3], state[4], state[5], 0, 0, 0]

# Generate true trajectory
true_states = odeint(robot_motion, x0, t)

# Add noise to velocity measurements
measurement_noise = 0.1
measurements = true_states[:, 3:] + np.random.normal(0, measurement_noise, (len(t), 3))

class RobotVisualizer:
    def __init__(self, width=800, height=600):
        pygame.init()
        
        # Set up display
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Robot Trajectory")
        
        # Add runway dimensions
        self.runway_length = 300  # Length in meters
        self.runway_width = 30    # Width in meters
        
        # Camera parameters
        self.camera_distance = 500
        self.camera_x = 0
        self.camera_y = 30
        self.camera_z = 100
        self.look_at = [0, 0, 0]
        self.up_vector = [0, 1, 0]
        
        # Mouse control state
        self.prev_mouse_pos = None
        self.mouse_button_down = False
        self.rotation_x = 30
        self.rotation_y = 0
        
        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up lighting
        glLight(GL_LIGHT0, GL_POSITION, (0, 1000, 0, 1))
        
        # Set up projection
        self.resize(width, height)
        
        # Add colors for different trajectories
        self.true_color = (1, 0.5, 0)      # Orange for true trajectory
        self.estimated_color = (0, 1, 0.5)  # Cyan for estimated trajectory
        self.measurement_color = (1, 0, 1)  # Magenta for measurements
        self.measurement_size = 5.0
        
        # Add inset view parameters
        self.inset_size = (200, 150)
        self.inset_position = (20, 20)
        self.inset_surface = pygame.Surface(self.inset_size)
        self.inset_texture = glGenTextures(1)
    
    def resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (width / height), 0.1, 2000.0)
        
    def draw_grid(self, size=500, step=50):
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0.5)
        
        for i in range(-size, size + step, step):
            # Draw lines parallel to X axis
            glVertex3f(-size, 0, i)
            glVertex3f(size, 0, i)
            # Draw lines parallel to Z axis
            glVertex3f(i, 0, -size)
            glVertex3f(i, 0, size)
            
        glEnd()
        
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
        
    def draw_robot(self, position, scale=20):
        x, y, z = position
        
        glPushMatrix()
        glTranslatef(x, y, z)
        
        # Draw robot body as a cube since we don't have GLUT
        glColor3f(0.7, 0.7, 0.7)
        self.draw_cube(5)
        
        # Draw robot coordinate frame
        glBegin(GL_LINES)
        # X axis
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(scale, 0, 0)
        # Y axis
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, scale, 0)
        # Z axis
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
        glColor3f(*color)
        glBegin(GL_LINE_STRIP)
        for point in points:
            glVertex3f(*point)
        glEnd()
        
    def draw_measurements(self, points):
        """Draw measurement points as small spheres"""
        glColor3f(*self.measurement_color)
        for point in points:
            glPushMatrix()
            glTranslatef(*point)
            # Draw point as a small cube since we're not using GLUT
            self.draw_cube(self.measurement_size)
            glPopMatrix()
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Check for escape key
                    return False
                
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
        
    def draw_aircraft_view(self, aircraft_position, aircraft_orientation):
        """Draw the view from aircraft's camera"""
        # Save current viewport and matrices
        glPushAttrib(GL_VIEWPORT_BIT)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        # Set up viewport for inset
        glViewport(self.inset_position[0], self.inset_position[1], 
                   self.inset_size[0], self.inset_size[1])
        
        # Set up camera projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.inset_size[0]/self.inset_size[1], 0.1, 2000.0)
        
        # Set up camera position/orientation
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Look along the trajectory (forward is -x direction since we start at negative x)
        gluLookAt(aircraft_position[0], aircraft_position[1], aircraft_position[2],  # Camera position
                  aircraft_position[0] + 10, aircraft_position[1], aircraft_position[2],  # Look forward along x-axis
                  0, 1, 0)  # Up vector
        
        # Draw the entire scene
        self.draw_grid()
        self.draw_axes()
        self.draw_runway()
        
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
        
        while running:
            running = self.handle_input()
            
            # Clear screen and set camera
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Set camera position
            cam_x = self.camera_distance * math.sin(math.radians(self.rotation_y)) * math.cos(math.radians(self.rotation_x))
            cam_y = self.camera_distance * math.sin(math.radians(self.rotation_x))
            cam_z = self.camera_distance * math.cos(math.radians(self.rotation_y)) * math.cos(math.radians(self.rotation_x))
            
            gluLookAt(cam_x + self.look_at[0], cam_y + self.look_at[1], cam_z + self.look_at[2],
                     *self.look_at,
                     *self.up_vector)
            
            # Draw scene
            self.draw_grid()
            self.draw_axes()
            self.draw_runway()
            
            # Draw true trajectory
            current_true = true_trajectory[:current_point+1]
            if len(current_true) > 0:
                self.draw_trajectory(current_true, self.true_color)
                self.draw_robot(current_true[-1], scale=20)
                
                # Add aircraft view after main scene is drawn
                aircraft_pos = current_true[-1]
                # Calculate orientation from velocity (if available) or use simple forward-facing orientation
                if len(current_true) > 1:
                    direction = current_true[-1] - current_true[-2]
                    aircraft_orientation = np.arctan2(direction[2], direction[0])
                else:
                    aircraft_orientation = 0
                self.draw_aircraft_view(aircraft_pos, aircraft_orientation)
            
            # Draw estimated trajectory
            current_estimated = estimated_trajectory[:current_point+1]
            if len(current_estimated) > 0:
                self.draw_trajectory(current_estimated, self.estimated_color)
                # Draw estimated robot position with smaller scale
                self.draw_robot(current_estimated[-1], scale=15)
            
            # Draw measurements if available
            if measurements is not None:
                self.draw_measurements(measurements)
            
            pygame.display.flip()
            current_point = (current_point + 1) % len(true_trajectory)
            clock.tick(60)
            
        pygame.quit()

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
measurement_noise = 10.0  # Adjust this value to control noise magnitude
noisy_measurements = true_trajectory[measurement_indices] + np.random.normal(0, measurement_noise, (len(measurement_indices), 3))

# Calculate velocity estimates from position deltas
velocity_estimates = np.zeros_like(true_trajectory)
velocity_estimates[1:] = (true_trajectory[1:] - true_trajectory[:-1]) / dt
velocity_estimates[0] = velocity_estimates[1]

# Create non-Gaussian noise with multiple components
velocity_noise_gaussian = 4.0  # Base Gaussian noise
velocity_noise_uniform = 2.0   # Uniform noise range
velocity_noise_random_walk = 0.1  # Random walk component
velocity_bias = np.array([0.5, -0.3, 0.2])  # Systematic bias

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

def ekf_update(state, P, velocity_estimate, Q, R, dt):
    # Predict step using provided noisy velocity estimate
    x, y, z, vx, vy, vz = state
    
    # Update velocities from estimate
    vx, vy, vz = velocity_estimate
    
    # Calculate Jacobian F at current state
    F = F_num(x, y, z, vx, vy, vz)
    F = np.array(F, dtype=float)
    
    # State transition with estimated velocities
    state_pred = state + np.array([vx, vy, vz, 0, 0, 0]) * dt
    state_pred[3:] = velocity_estimate  # Update velocity states
    
    # Covariance prediction
    P_pred = F @ P @ F.T + Q
    
    # No measurement updates for now
    state = state_pred
    P = P_pred
    
    return state, P

# Run EKF with velocity estimates
estimated_trajectory = np.zeros_like(true_trajectory)
state = np.array([x[0], y[0], z[0], *noisy_velocities[0]])  # Initial state with first velocity estimate
P = np.eye(6) * 1.0  # Initial state covariance
Q = np.eye(6) * 0.1  # Process noise covariance
R = np.eye(3) * measurement_noise  # Measurement noise covariance

for i in range(len(true_trajectory)):
    # Use noisy velocity estimate for prediction
    state, P = ekf_update(state, P, noisy_velocities[i], Q, R, dt)
    estimated_trajectory[i] = state[:3]

# Create and run visualizer with both trajectories
visualizer = RobotVisualizer()
visualizer.run_simulation(true_trajectory, estimated_trajectory, noisy_measurements)
