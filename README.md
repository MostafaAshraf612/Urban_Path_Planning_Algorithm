# 🚗 Motion Planning for Self-Driving Cars - Final Project 🚦

This repository contains the implementation and solution for the Final Project of the Coursera specialization course [Motion Planning for Self-Driving Cars](https://www.coursera.org/learn/motion-planning-self-driving-cars/programming/wiGwg/course-4-final-project). The project demonstrates the development of a hierarchical motion planner designed to navigate an autonomous vehicle safely and efficiently through complex urban driving scenarios.

## 📋 Project Overview

The objective of this project is to architect and implement a comprehensive hierarchical motion planning system incorporating behavior and trajectory planning modules. This system addresses diverse driving tasks including obstacle avoidance, lane keeping, and intersection navigation within a high-fidelity simulated environment.

The proposed planner comprises:
- 🗺️ Mission Planning: Strategic route selection based on global path information
- 🤖 Behavior Planning: Finite State Machine (FSM) to determine safe and contextually appropriate maneuvers such as lane changes, stops, and car following
- 🛣️ Trajectory Generation: Dynamic trajectory synthesis that ensures smooth, feasible paths while adhering to vehicle kinematics and safety constraints

Key testing scenarios involve:
- 🚧 Navigating around static obstacles like parked vehicles obstructing travel lanes
- 🚘 Maintaining safe following distances behind lead vehicles
- 🛑 Executing intersection behaviors aligned with traffic regulations and right-of-way protocols

## 🔗 Course and Project Resources

- [📚 Official Course Page: Motion Planning for Self-Driving Cars](https://www.coursera.org/learn/motion-planning-self-driving-cars/programming/wiGwg/course-4-final-project)
- [📝 Final Project Specifications and Requirements](https://www.coursera.org/learn/motion-planning-self-driving-cars/programming/wiGwg/course-4-final-project)

## 🗂 Repository Structure

---

## ✨ Key Features and Highlights

- Robust hierarchical planning framework integrating mission-level navigation with local behavior and trajectory generation
- Well-defined Finite State Machine facilitating reliable behavior decisions in dynamic traffic environments
- Generation of high-quality, smooth vehicle trajectories respecting acceleration, jerk limitations, and dynamic constraints
- Collision detection and avoidance performed using occupancy grid mapping
- Modular and extensible design targeted for simulation within the CARLA autonomous driving simulator

## 🛠 Technology Stack

- Python programming language utilized for core algorithm implementation and simulation orchestration
- CARLA simulator employed for realistic urban driving scenario evaluation
- Interactive Jupyter Notebooks providing detailed visualizations and step-by-step algorithm demonstrations
- Utilization of scientific computing libraries including NumPy and Matplotlib to support data processing and visualization

## ▶️ Installation and Execution Guide

1. Clone this repository to your local environment
2. Install the required dependencies as specified in `requirements.txt`
3. Launch the CARLA simulator and load the provided scenario configurations from the `data/` folder
4. Execute the planner by running the primary scripts located in the `src/` directory or using the Jupyter notebook demonstration file
5. Observe vehicle behavior and planned trajectories within the simulator's visualization interface

## 📝 Running Instructions

- Initiate the CARLA simulation platform and load predetermined urban driving scenarios
- Execute the motion planner script; it will establish connections with the simulator and begin planning operations
- Visual outputs including planned paths, state machine transitions, and vehicle control commands will be rendered live in the simulator
- Modify configuration settings to examine planner performance under varying traffic densities and obstacle layouts

## 🎓 Learning Outcomes and Professional Skills Demonstrated

- Design and development of an autonomous vehicle motion planning pipeline conforming to realistic operational constraints
- Application of finite state machines for high-level behavior decision making in dynamic environments
- Generation of safe, feasible trajectories that meet kinematic and dynamic vehicle constraints
- Proficiency in simulation-based validation and iterative algorithm refinement for self-driving car technologies

## 📜 Licensing

This repository and its contents are distributed solely for academic and educational use, specifically in accordance with the Coursera Motion Planning specialization course guidelines.

---

Developed as part of the Self-Driving Cars Specialization on Coursera. 🚀
