# Cymbal

![3D EKF Visualization](media/header.gif)

<div align="center">

**A 3D Extended Kalman Filter Visualization for Aircraft Trajectory Tracking**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Real-time pose estimation, trajectory tracking, and sensor fusion visualization*

</div>

---

## Overview

**Cymbal** is an advanced visualization tool for demonstrating Extended Kalman Filter (EKF) algorithms in 3D space. It simulates an aircraft during landing approach, tracking position and orientation through noisy sensor measurements using Perspective-n-Point (PnP) computer vision algorithms.

Built with scientific computing and real-time 3D graphics, Cymbal serves as both an educational platform for learning Kalman filtering and a research testbed for robotics and drone navigation systems.

---

## Features

### Core Capabilities

- **13-Dimensional State Estimation**
  - Position (x, y, z)
  - Quaternion orientation (qw, qx, qy, qz)
  - Linear velocity (vx, vy, vz)
  - Angular velocity (wx, wy, wz)

- **Extended Kalman Filter**
  - Symbolic Jacobian computation using SymPy
  - Real-time prediction and update steps
  - Quaternion-based orientation handling
  - Configurable process and measurement noise

- **Perspective-n-Point (PnP) Tracking**
  - OpenCV-based Epnp solver
  - 3D to 2D projection with configurable focal length
  - Point validation for camera occlusions

- **Real-Time 3D Visualization**
  - OpenGL rendering with custom shaders
  - Multi-viewport display (third-person + two inset views)
  - Procedural terrain generation
  - Interactive camera controls with auto-orbit

### Simulation Environment

- Realistic 3-degree glide slope landing approach
- Configurable sensor noise injection
- Frame recording for animation export
- Performance monitoring with frame timing analysis

---

## Installation

### Prerequisites

- Python 3.9 or higher
- OpenGL 3.3+ compatible graphics driver

### Setup with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/cymbal.git
cd cymbal

# Install dependencies and create virtual environment
uv sync
```

### Manual Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Basic Usage

Run the main visualization:

```bash
python core.py
```

### Running Tests

Execute the test suite:

```bash
python ekf_pnp_tests.py
```

### Exporting Animation

1. Press `C` to start/stop recording
2. Frames are saved to `output_frames/`
3. Convert frames to GIF using the utility:

```python
from media.gif import frames_to_gif

frames_to_gif(
    input_dir="output_frames",
    output="animation.gif",
    fps=30,
    duration=0.5
)
```

---

## Controls

| Key | Action |
|-----|--------|
| `Mouse Left` | Rotate camera |
| `Mouse Right` | Pan camera |
| `Mouse Wheel` | Zoom in/out |
| `R` | Toggle runway display |
| `G` | Toggle grid display |
| `X` | Toggle axes display |
| `C` | Toggle frame recording |
| `ESC` | Exit simulation |

---

## Technical Details

### State Vector

The EKF maintains a 13-dimensional state vector:

```
s = [cx, cy, cz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
```

- **c**: Camera position in world coordinates
- **q**: Orientation as a normalized quaternion [w, x, y, z]
- **v**: Linear velocity
- **w**: Angular velocity

### Motion Model

Constant velocity model with quaternion kinematics for rotation:

```
c_{t+1} = c_t + v_t * dt
q_{t+1} = q_t ⊗ Δq(ω_t * dt)
v_{t+1} = v_t
ω_{t+1} = ω_t
```

### Measurement Model

Pinhole camera projection for PnP:

```
x_c = R * (p_w - c)
x_i = f * [x_c[0]/x_c[2], x_c[1]/x_c[2]]
```

Where `f` is the focal length and `R` is the rotation matrix derived from the quaternion.

### Jacobian Computation

Jacobian matrices are computed symbolically using SymPy for accurate linearization:

```python
F = ∂f/∂s  # State transition Jacobian
H = ∂h/∂s  # Measurement Jacobian
```

---

## Architecture

```
cymbal/
├── core.py              # Main simulation and visualization (1002 lines)
├── ekf_pnp.py          # EKF-PnP implementation (255 lines)
├── ekf_pnp_tests.py    # Test suite (588 lines)
├── media/
│   └── gif.py          # Frame-to-GIF conversion
├── main.py             # Entry point
├── pyproject.toml      # Project configuration
└── requirements.txt    # Python dependencies
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | - | Numerical computations |
| SciPy | - | Integration, spatial transforms |
| SymPy | - | Symbolic Jacobian computation |
| PyOpenGL | - | 3D graphics rendering |
| Pygame | - | Window management, input |
| OpenCV | - | PnP computer vision |
| Matplotlib | - | Data visualization |
| PyFlame | - | Performance profiling |

---

## Configuration

Key parameters can be adjusted in `core.py`:

```python
# Simulation
dt = 0.1                    # Time step
t = np.arange(0, 60, dt)    # Duration

# Camera
focal_length = 800          # Pixels

# Process Noise
sigma_a = 1.0               # Linear acceleration
sigma_alpha = 0.1           # Angular acceleration

# Measurement Noise
position_noise = 10.0       # Position std dev
orientation_noise = 0.05    # Orientation std dev
```

---

## Algorithm Details

### Extended Kalman Filter

The EKF consists of two steps:

**Prediction:**
```
ŝₜ₊₁|ₜ = f(ŝₜ|ₜ)
Pₜ₊₁|ₜ = F Pₜ|ₜ Fᵀ + Q
```

**Update:**
```
K = P Hᵀ (H P Hᵀ + R)⁻¹
ŝₜ₊₁|ₜ₊₁ = ŝ₊₁|ₜ + K(z - h(ŝₜ₊₁|ₜ))
Pₜ₊₁|ₜ₊₁ = (I - KH)Pₜ₊₁|ₜ
```

### Quaternion Kinematics

Orientation updates use quaternion multiplication:

```python
Δq = [cos(θ/2), sin(θ/2) * n_x, sin(θ/2) * n_y, sin(θ/2) * n_z]
q_new = q_old ⊗ Δq
```

---

## Examples

### Landing Approach Trajectory

The aircraft follows a 3-degree glide slope from (-1908m, 100m) to the origin:

```python
glide_slope = 3  # degrees
initial_altitude = 100  # meters
initial_x = -(initial_altitude / tan(3°))  # ≈ -1908m
```

### Noise Injection

Sensor measurements include realistic noise:

```python
position_measurements += N(0, 10.0)      # 10m std deviation
orientation_measurements += N(0, 0.05)   # ~3° std deviation
```

---

## Performance

Frame timing breakdown (typical):

| Component | Time (ms) |
|-----------|-----------|
| Input handling | ~0.1 |
| GL Setup | ~0.5 |
| Scene rendering | ~2.0 |
| EKF computation | ~0.3 |
| Trajectory drawing | ~0.5 |
| Display flip | ~1.0 |
| **Total** | **~5-10** |

Target: 30 FPS (33ms per frame)

---

## Use Cases

- **Education**: Teaching Kalman filters and state estimation
- **Research**: Testing new filtering algorithms
- **Development**: Prototyping robotics/navigation systems
- **Benchmarking**: Performance testing for real-time visualization

---

## References

- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
- Julier, S. J., & Uhlmann, J. K. (1997). "A New Extension of the Kalman Filter to Nonlinear Systems"
- Lepetit, V., et al. (2009). "EPnP: An Accurate O(n) Solution to the PnP Problem"

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
