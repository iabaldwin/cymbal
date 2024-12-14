# 3D Extended Kalman Filter Visualization

![3D EKF Visualization](media/header.gif)

A real-time 3D visualization tool for Extended Kalman Filter (EKF) trajectory estimation, implemented in Python using OpenGL and Pygame.

## Features

- Real-time 3D visualization of true trajectory and EKF estimates
- Interactive camera controls with auto-orbit functionality
- Multiple viewports showing both third-person and first-person perspectives
- Procedurally generated terrain with runway
- OpenGL-based rendering with modern shader support
- Frame recording capabilities for creating animations
- Realistic aircraft landing approach simulation

## Requirements

- Python 3.7+
- NumPy
- SciPy
- PyOpenGL
- Pygame
- SymPy
- Pillow (PIL)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/3d-ekf-visualization.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python core.py
```

### Controls

- **Left Mouse Button**: Rotate camera
- **Right Mouse Button**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **R**: Toggle runway visibility
- **G**: Toggle terrain grid visibility
- **X**: Toggle coordinate axes visibility
- **C**: Toggle frame recording
- **ESC**: Exit application

## Technical Details

The visualization demonstrates an Extended Kalman Filter tracking a simulated aircraft on a landing approach. The state vector includes:

- Position (x, y, z)
- Velocity (vx, vy, vz)
- Orientation (quaternion: qw, qx, qy, qz)

The simulation includes:
- 3-degree glide slope approach
- Realistic measurement noise
- Process model with constant velocity assumption
- Sparse measurements (every 20 timesteps)

## Implementation Details

- OpenGL-based rendering with modern shader pipeline
- Real-time EKF state estimation
- Multiple coordinate frame transformations
- Quaternion-based orientation representation
- Procedural terrain generation
- Multi-viewport rendering