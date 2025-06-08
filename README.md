# 🌬️ Thermoflow2D Solver Pro

A 2D laminar fluid flow and heat transfer simulation tool with a modern graphical user interface (GUI), built using `Tkinter` and `ttkbootstrap`. This solver uses the vorticity-streamfunction formulation of the incompressible Navier-Stokes equations, coupled with the energy equation.

---

## 📁 Folder Structure
Thermoflow2D-Solver-Pro/
├── GUI.py # Main Python GUI application
└── User Manual.pdf # User guide with instructions and screenshots

---

## 💡 Features

- ✅ Interactive GUI with real-time simulation controls
- 🎨 Light/Dark theme toggle via `ttkbootstrap`
- 💾 Save and load simulation configurations (`.json`)
- 📈 Built-in animation of velocity fields and temperature distribution
- 📊 Line plots of horizontal/vertical velocities and temperature profiles
- 🧮 Solves:
  - 2D incompressible Navier-Stokes (vorticity-streamfunction)
  - 2D energy equation (heat transfer)

---

## 🔧 Numerical Methods

| Component           | Method                |
|---------------------|------------------------|
| Time Integration    | Explicit Euler         |
| Spatial Derivatives | Second-order Central Difference |
| Poisson Solver      | Successive Over-Relaxation (SOR) |
| Advection-Diffusion | Explicit scheme        |

---

## ▶️ How to Run

### 🐍 Requirements

- Python ≥ 3.8
- Packages:
  - `numpy`
  - `matplotlib`
  - `ttkbootstrap`

### 💻 Run the GUI

```bash
python GUI.py
📘 User Manual
See User Manual.pdf for step-by-step instructions including:

How to set up flow and thermal parameters

How to interpret the plots

How to use Neumann/Dirichlet boundary conditions

How to save/load configurations
🖼️ Interface Preview
Animation window with 6 subplots:

Velocity magnitude & streamlines

u / v component plots

Temperature field

Midline velocity & temperature profiles
