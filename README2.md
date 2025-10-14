# 🚗 Urban Path Planning Algorithm  
**Coursera Self-Driving Cars Specialization – Motion Planning Course**  
**Developer:** [Mostafa Ashraf El Sayed](https://www.linkedin.com/in/mostafa-ashraf-612)

![License: Academic Use](https://img.shields.io/badge/License-Academic_Use-lightgrey.svg)
![Language: Python](https://img.shields.io/badge/Language-Python3-blue.svg)
![Status: Completed](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup & Execution](#setup--execution)
- [Algorithm Summary](#algorithm-summary)
- [Decision Logic](#decision-logic)
- [Results & Demonstration](#results--demonstration)
- [Performance Metrics](#performance-metrics)
- [License](#license)
- [Contact](#contact)

---

## 📌 Overview

An intelligent **Python-based motion planner** designed for autonomous vehicles navigating complex urban environments.  
This system integrates **behavior planning**, **trajectory generation**, and **collision avoidance** to produce safe, smooth, and dynamically feasible paths.

Developed as part of the **Coursera Self-Driving Cars Specialization**, the planner demonstrates hierarchical motion planning using FSMs, spline-based trajectory generation, and simulation-based validation.

---

## ✨ Features

- **Hierarchical Planning:** Combines global routing with local decision-making  
- **Finite State Machine (FSM):** Manages lane keeping, lane changes, and stopping behavior  
- **Smooth Trajectory Generation:** Produces jerk-limited, kinematically feasible paths  
- **Collision Avoidance:** Detects and avoids both static and dynamic obstacles  
- **Modular Design:** Clean separation of behavior, trajectory, and collision logic  
- **Simulation-Validated:** Tested in realistic urban scenarios using CARLA simulator

---

## 🧠 Architecture

1. **Perception Input** – Processes waypoint and sensor data  
2. **Behavior Planning** – FSM selects driving state based on traffic context  
3. **Trajectory Generation** – Generates smooth, safe paths using splines  
4. **Collision Checking** – Validates trajectory safety against obstacles  
5. **Control Output** – Sends trajectory to simulator or controller  

---

## 📁 Repository Structure

```
Course4FinalProject/
├── __init__.py
├── behavioural_planner.py         # Behavior planning logic (FSM)
├── collision_checker.py           # Collision detection for static/dynamic obstacles
├── controller2d.py                # Low-level vehicle control
├── course4_waypoints.txt          # Waypoint data for simulation
├── live_plotter.py                # Real-time plotting utility
├── local_planner.py               # Local trajectory generation
├── module_7.py                    # Main execution script
├── options.cfg                    # Configuration file
├── parked_vehicle_params.txt      # Parameters for parked vehicle detection
├── path_optimizer.py              # Path smoothing and optimization
├── stop_sign_params.txt           # Parameters for stop sign behavior
├── velocity_planner.py            # Speed profile generation
├── utils.py                       # Helper functions
├── README.md                      # Project documentation
│
├── controller_output/
│   ├── collision.txt
│   ├── collision_count.txt
│   ├── trajectory.txt
│
├── assets/
│   └── Urban_Planning_demo_preview.gif  # Demo visualization
│
└── __pycache__/                   # Python bytecode cache
```

---

## 🚀 Setup & Execution

To execute the Urban Path Planning Algorithm within the CARLA Simulator environment, follow the steps below in sequence:

---

### 📥 Step 1: Clone the Project Repository

```bash
git clone https://github.com/MostafaAshraf612/Urban_Path_Planning_Algorithm.git
```

---

### 📁 Step 2: Integrate with CARLA Simulator

Move the cloned repository into the `PythonClient` directory of your CARLA installation:

```bash
mv Urban_Path_Planning_Algorithm /path/to/CARLA/PythonClient/
```

Replace `/path/to/CARLA/` with the actual path to your CARLA root directory.

---

### 🐍 Step 3: Create and Activate a Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
cd /path/to/CARLA/PythonClient/Urban_Path_Planning_Algorithm
python3 -m venv urban_env
source urban_env/bin/activate
```

---

### 📦 Step 4: Install Python Dependencies

Install the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 🎮 Step 5: Launch the CARLA Simulator

In a separate terminal, navigate to the CARLA root directory and launch the simulator with the Course 4 map:

```bash
./CarlaUE4.sh /Game/Maps/Course4 -windowed -ResX=1500 -ResY=1000 -carla-server -opengl -benchmark -fps=30
```

---

### 🚗 Step 6: Execute the Urban Path Planner

Return to the project directory and run the planner:

```bash
cd /path/to/CARLA/PythonClient/Urban_Path_Planning_Algorithm
python module_7.py
```

> ✅ Ensure that all project files remain in the root of the `Urban_Path_Planning_Algorithm` folder to maintain compatibility with the execution pipeline.

---

## 📈 Algorithm Summary

The planner continuously evaluates the driving environment to decide whether to:

- **Maintain the current lane**  
- **Perform a lane change**  
- **Adjust the vehicle’s speed**

It generates **smooth, feasible trajectories** that respect vehicle dynamics and avoid collisions with both static and dynamic obstacles.

---

## 🧭 Decision Logic

- **Monitor surrounding vehicles** using sensor and waypoint data  
- **Evaluate adjacent lanes** for safe lane change opportunities  
- **Execute lane change** if feasible; otherwise, **smoothly decelerate**  
- **Continuously update trajectory** to adapt to real-time traffic and vehicle dynamics  

---

## 🎥 Results & Demonstration

The planner successfully performs:

- Safe lane keeping and smooth lane changes  
- Speed regulation within acceleration and jerk limits  
- Obstacle avoidance in dynamic urban scenarios

📹 **Demo Preview:**  
<p align="center">
  <a href="assets/Urban_Planning_demo_preview.gif">
    <img src="assets/Urban_Planning_demo_preview.gif" 
         alt="Urban Planning Demo" 
         width="65%" 
         style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
  </a>
</p>

<p align="center">
  <strong>View Full Demonstration:</strong><br>
  <a href="https://onedrive.live.com/?<your_file_link>" target="_blank">
    ➡️ Watch on OneDrive
  </a>
</p>

---

## ✅ Performance Metrics

| 🔍 **Metric**             | 📊 **Value**     | 📝 **Description**                                |
|---------------------------|------------------|---------------------------------------------------|
| **Target Speed**          | 30 km/h          | Maintains safe, consistent speed in urban areas   |
| **Maximum Acceleration**  | 2.5 m/s²         | Respects dynamic limits for comfort and control   |
| **Maximum Jerk**          | < 10 m/s³        | Ensures passenger comfort and stability           |
| **Lane Change Time**      | < 2 seconds      | Smooth transitions between lanes                  |
| **Collisions**            | 0                | No collisions during validation scenarios         |
| **Path Smoothness**       | High             | Spline interpolation ensures smooth motion        |

---

## 📄 License

This repository is provided for **academic and educational purposes** under Coursera course guidelines.

---

## 📬 Contact

For technical inquiries or collaboration opportunities:

**Mostafa Ashraf El Sayed**  
🔗 [LinkedIn](https://www.linkedin.com/in/mostafa-ashraf-612)  
💻 [GitHub](https://github.com/MostafaAshraf612)  
📧 [mostafashrafelsayed612@gmail.com](mailto:mostafashrafelsayed612@gmail.com)
