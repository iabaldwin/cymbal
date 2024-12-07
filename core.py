import numpy as np
import sympy as sp
from scipy.integrate import odeint
import pygame
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
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D Robot Trajectory")
        
        # Camera parameters
        self.camera_distance = 500
        self.camera_angle_x = math.pi / 4
        self.camera_angle_y = math.pi / 6
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
    def project_3d_to_2d(self, point3d):
        # Rotate point
        x, y, z = point3d
        # Rotate around Y axis
        x_rot = x * math.cos(self.camera_angle_y) + z * math.sin(self.camera_angle_y)
        z_rot = -x * math.sin(self.camera_angle_y) + z * math.cos(self.camera_angle_y)
        # Rotate around X axis
        y_rot = y * math.cos(self.camera_angle_x) - z_rot * math.sin(self.camera_angle_x)
        z_rot_2 = y * math.sin(self.camera_angle_x) + z_rot * math.cos(self.camera_angle_x)
        
        # Project to 2D
        scale = self.camera_distance / (self.camera_distance + z_rot_2)
        x2d = self.width/2 + x_rot * scale
        y2d = self.height/2 + y_rot * scale
        
        return int(x2d), int(y2d)
    
    def draw_axes(self):
        # Draw coordinate axes
        origin = self.project_3d_to_2d((0, 0, 0))
        x_end = self.project_3d_to_2d((100, 0, 0))
        y_end = self.project_3d_to_2d((0, 100, 0))
        z_end = self.project_3d_to_2d((0, 0, 100))
        
        pygame.draw.line(self.screen, self.RED, origin, x_end, 2)
        pygame.draw.line(self.screen, self.GREEN, origin, y_end, 2)
        pygame.draw.line(self.screen, self.BLUE, origin, z_end, 2)
    
    def draw_trajectory(self, points, color=(255, 255, 0)):
        # Draw trajectory line
        projected_points = [self.project_3d_to_2d(point) for point in points]
        if len(projected_points) > 1:
            pygame.draw.lines(self.screen, color, False, projected_points, 2)
        
        # Draw current position
        if projected_points:
            pygame.draw.circle(self.screen, (255, 0, 0), projected_points[-1], 5)
    
    def run_simulation(self, trajectory):
        clock = pygame.time.Clock()
        running = True
        current_point = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.camera_angle_y -= 0.1
                    elif event.key == pygame.K_RIGHT:
                        self.camera_angle_y += 0.1
                    elif event.key == pygame.K_UP:
                        self.camera_angle_x -= 0.1
                    elif event.key == pygame.K_DOWN:
                        self.camera_angle_x += 0.1
            
            self.screen.fill((0, 0, 0))
            self.draw_axes()
            
            # Draw trajectory up to current point
            current_trajectory = trajectory[:current_point+1]
            self.draw_trajectory(current_trajectory)
            
            pygame.display.flip()
            
            current_point = (current_point + 1) % len(trajectory)
            clock.tick(30)
        
        pygame.quit()

# Generate a more interesting trajectory (spiral motion)
t = np.arange(0, 20, 0.1)
radius = 100
frequency = 0.5
ascent_rate = 10

x = radius * np.cos(frequency * t)
y = radius * np.sin(frequency * t)
z = ascent_rate * t

trajectory = np.column_stack((x, y, z))

# Create and run visualizer
visualizer = RobotVisualizer()
visualizer.run_simulation(trajectory)