import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm, animation
import matplotlib.contour
import json
import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class CFDSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermoflow2D Solver Pro")
        # Initialize with light theme
        self.current_theme = 'litera'
        self.style = ttk.Style(theme=self.current_theme)
        # Initialize all tk variables
        self.lx_var = tk.DoubleVar()
        self.ly_var = tk.DoubleVar()
        self.dx_var = tk.DoubleVar()
        self.dy_var = tk.DoubleVar()
        self.re_var = tk.DoubleVar()
        self.dt_var = tk.DoubleVar()
        self.nt_var = tk.IntVar()
        self.pr_var = tk.DoubleVar()
        self.create_widgets()
        self.fig = None
        self.canvas = None
        self.anim = None

    def create_widgets(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10", style='primary.TFrame')
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Input and plot frames
        self.input_frame = ttk.Frame(self.main_frame, padding="5", style='light.TFrame')
        self.input_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=10)
        self.plot_frame = ttk.Frame(self.main_frame, padding="5", style='light.TFrame')
        self.plot_frame.grid(row=0, column=1, sticky=(tk.E, tk.N, tk.S, tk.W), padx=10)
        self.main_frame.columnconfigure(1, weight=3)

        # Notebook
        self.notebook = ttk.Notebook(self.input_frame, style='primary.TNotebook')
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=5)

        # Flow parameters frame
        flow_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(flow_frame, text="Flow Parameters")
        self.create_flow_widgets(flow_frame)

        # Thermal parameters frame
        thermal_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(thermal_frame, text="Thermal Parameters")
        self.create_thermal_widgets(thermal_frame)

        # Console
        self.console = scrolledtext.ScrolledText(
            self.input_frame, height=10, width=50, 
            font=('Arial', 10), wrap=tk.WORD,
            borderwidth=2, relief='flat', 
            background='#f8f9fa' if self.current_theme == 'litera' else '#343a40'
        )
        self.console.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))
        self.console.insert(tk.END, "Simulation output will appear here...\n")

        # Button frame
        self.button_frame = ttk.Frame(self.input_frame)
        self.button_frame.grid(row=2, column=0, pady=10, sticky=tk.W)
        ttk.Button(
            self.button_frame, text="Run Simulation", 
            command=self.run_simulation, 
            style='primary.TButton',
            padding=8
        ).grid(row=0, column=0, padx=5)
        ttk.Button(
            self.button_frame, text="Save Config", 
            command=self.save_config, 
            style='success.TButton',
            padding=8
        ).grid(row=0, column=1, padx=5)
        ttk.Button(
            self.button_frame, text="Load Config", 
            command=self.load_config, 
            style='info.TButton',
            padding=8
        ).grid(row=0, column=2, padx=5)
        # Add theme toggle button
        self.theme_button = ttk.Button(
            self.button_frame, text="Switch to Dark Theme", 
            command=self.toggle_theme, 
            style='secondary.TButton',
            padding=8
        )
        self.theme_button.grid(row=0, column=3, padx=5)

        # Plot area label
        self.plot_label = ttk.Label(
            self.plot_frame, 
            text="Animation will appear here after running simulation",
            style='primary.TLabel',
            font=('Arial', 12, 'italic'),
            anchor='center'
        )
        self.plot_label.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

    def toggle_theme(self):
        # Toggle between light and dark themes
        if self.current_theme == 'litera':
            self.current_theme = 'darkly'
            self.theme_button.configure(text="Switch to Light Theme")
            self.console.configure(background='#343a40', foreground='#ffffff')
        else:
            self.current_theme = 'litera'
            self.theme_button.configure(text="Switch to Dark Theme")
            self.console.configure(background='#f8f9fa', foreground='#000000')
        
        # Update the style theme
        self.style.theme_use(self.current_theme)
        # Update styles for better visibility in dark mode
        self.style.configure('Custom.TEntry', 
                           fieldbackground='#343a40' if self.current_theme == 'darkly' else '#ffffff',
                           foreground='#ffffff' if self.current_theme == 'darkly' else '#000000')

    def create_flow_widgets(self, frame):
        # Style configuration
        entry_style = ttk.Style()
        entry_style.configure('Custom.TEntry', 
                            padding=5, 
                            font=('Arial', 10),
                            fieldbackground='#343a40' if self.current_theme == 'darkly' else '#ffffff',
                            foreground='#ffffff' if self.current_theme == 'darkly' else '#000000')
        
        params = [
            ("Domain Length (Lx):", self.lx_var, 1.0),
            ("Domain Width (Ly):", self.ly_var, 1.0),
            ("Target dx:", self.dx_var, 0.01),
            ("Target dy:", self.dy_var, 0.01),
            ("Reynolds Number (Re):", self.re_var, 100),
            ("Time Step (dt):", self.dt_var, 0.0005),
            ("Number of Steps (nt):", self.nt_var, 1000)
        ]
        
        for i, (label, var, default) in enumerate(params):
            ttk.Label(frame, text=label, style='primary.TLabel').grid(row=i, column=0, sticky=tk.W, pady=2)
            var.set(default)
            ttk.Entry(frame, textvariable=var, style='Custom.TEntry').grid(row=i, column=1, pady=2, padx=5)

        ttk.Label(frame, text="Velocity Boundary Conditions", style='primary.TLabel', font=('Arial', 11, 'bold')).grid(row=7, column=0, columnspan=2, pady=10)
        
        boundaries = ["left", "right", "top", "bottom"]
        self.u_vars = {}
        self.v_vars = {}
        self.velocity_bc_vars = {}
        
        boundary_labels = {
            "left": "Left",
            "right": "Right",
            "top": "Top",
            "bottom": "Bottom"
        }
        
        for i, boundary in enumerate(boundaries):
            display_label = boundary_labels[boundary]
            ttk.Label(frame, text=f"{display_label} U:").grid(row=8+i, column=0, sticky=tk.W, pady=2)
            self.u_vars[boundary] = tk.DoubleVar(value=1.0 if boundary == "top" else 0.0)
            ttk.Entry(frame, textvariable=self.u_vars[boundary], style='Custom.TEntry').grid(row=8+i, column=1, pady=2)

            ttk.Label(frame, text=f"{display_label} V:").grid(row=8+i, column=2, sticky=tk.W, pady=2)
            self.v_vars[boundary] = tk.DoubleVar(value=0.0)
            ttk.Entry(frame, textvariable=self.v_vars[boundary], style='Custom.TEntry').grid(row=8+i, column=3, pady=2)

            self.velocity_bc_vars[boundary] = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                frame, 
                text=f"{display_label} Neumann", 
                variable=self.velocity_bc_vars[boundary],
                style='primary.TCheckbutton'
            ).grid(row=8+i, column=4, padx=5)

    def create_thermal_widgets(self, frame):
        ttk.Label(frame, text="Prandtl Number (Pr):", style='primary.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.pr_var.set(0.71)
        ttk.Entry(frame, textvariable=self.pr_var, style='Custom.TEntry').grid(row=0, column=1, pady=2, padx=5)

        ttk.Label(frame, text="Temperature Boundary Conditions", style='primary.TLabel', font=('Arial', 11, 'bold')).grid(row=1, column=0, columnspan=3, pady=10)
        
        boundaries = ["left", "right", "top", "bottom"]
        self.t_vars = {}
        self.temperature_bc_vars = {}
        
        boundary_labels = {
            "left": "Left",
            "right": "Right",
            "top": "Top",
            "bottom": "Bottom"
        }
        
        default_temps = {
            "left": 300.0,
            "right": 270.0,
            "top": 300.0,
            "bottom": 30.0
        }
        
        for i, boundary in enumerate(boundaries):
            display_label = boundary_labels[boundary]
            ttk.Label(frame, text=f"{display_label} T (K):").grid(row=2+i, column=0, sticky=tk.W, pady=2)
            self.t_vars[boundary] = tk.DoubleVar(value=default_temps[boundary])
            ttk.Entry(frame, textvariable=self.t_vars[boundary], style='Custom.TEntry').grid(row=2+i, column=1, pady=2)

            self.temperature_bc_vars[boundary] = tk.BooleanVar(value=boundary != "top")
            ttk.Checkbutton(
                frame, 
                text=f"{display_label} Neumann", 
                variable=self.temperature_bc_vars[boundary],
                style='primary.TCheckbutton'
            ).grid(row=2+i, column=2, padx=5)

    def save_config(self):
        config = {
            "Lx": self.lx_var.get(),
            "Ly": self.ly_var.get(),
            "dx": self.dx_var.get(),
            "dy": self.dy_var.get(),
            "Re": self.re_var.get(),
            "dt": self.dt_var.get(),
            "nt": self.nt_var.get(),
            "Pr": self.pr_var.get(),
            "velocity": {
                boundary: {
                    "U": self.u_vars[boundary].get(),
                    "V": self.v_vars[boundary].get(),
                    "Neumann": self.velocity_bc_vars[boundary].get()
                } for boundary in ["left", "right", "top", "bottom"]
            },
            "temperature": {
                boundary: {
                    "T": self.t_vars[boundary].get(),
                    "Neumann": self.temperature_bc_vars[boundary].get()
                } for boundary in ["left", "right", "top", "bottom"]
            }
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)
            self.console.insert(tk.END, f"Configuration saved to {file_path}\n")

    def load_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, "r") as f:
                    config = json.load(f)
                self.lx_var.set(config["Lx"])
                self.ly_var.set(config["Ly"])
                self.dx_var.set(config["dx"])
                self.dy_var.set(config["dy"])
                self.re_var.set(config["Re"])
                self.dt_var.set(config["dt"])
                self.nt_var.set(config["nt"])
                self.pr_var.set(config["Pr"])
                for boundary in ["left", "right", "top", "bottom"]:
                    self.u_vars[boundary].set(config["velocity"][boundary]["U"])
                    self.v_vars[boundary].set(config["velocity"][boundary]["V"])
                    self.velocity_bc_vars[boundary].set(config["velocity"][boundary]["Neumann"])
                    self.t_vars[boundary].set(config["temperature"][boundary]["T"])
                    self.temperature_bc_vars[boundary].set(config["temperature"][boundary]["Neumann"])
                self.console.insert(tk.END, f"Configuration loaded from {file_path}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")
                self.console.insert(tk.END, f"Error loading config: {e}\n")

    def run_simulation(self):
        self.console.delete(1.0, tk.END)
        try:
            Lx = self.lx_var.get()
            Ly = self.ly_var.get()
            dx_target = self.dx_var.get()
            dy_target = self.dy_var.get()
            nx = int(np.ceil(Lx / dx_target)) + 1
            ny = int(np.ceil(Ly / dy_target)) + 1
            Re = self.re_var.get()
            dt = self.dt_var.get()
            nt = self.nt_var.get()
            Pr = self.pr_var.get()

            U_left = self.u_vars["left"].get()
            U_right = self.u_vars["right"].get()
            U_top = self.u_vars["top"].get()
            U_bottom = self.u_vars["bottom"].get()
            V_left = self.v_vars["left"].get()
            V_right = self.v_vars["right"].get()
            V_top = self.v_vars["top"].get()
            V_bottom = self.v_vars["bottom"].get()
            velocity_bc_left = self.velocity_bc_vars["left"].get()
            velocity_bc_right = self.velocity_bc_vars["right"].get()
            velocity_bc_top = self.velocity_bc_vars["top"].get()
            velocity_bc_bottom = self.velocity_bc_vars["bottom"].get()

            T_left = self.t_vars["left"].get()
            T_right = self.t_vars["right"].get()
            T_top = self.t_vars["top"].get()
            T_bottom = self.t_vars["bottom"].get()
            temperature_bc_left = self.temperature_bc_vars["left"].get()
            temperature_bc_right = self.temperature_bc_vars["right"].get()
            temperature_bc_top = self.temperature_bc_vars["top"].get()
            temperature_bc_bottom = self.temperature_bc_vars["bottom"].get()

            U_default = 1.0
            velocities = [U_left, U_right, U_top, U_bottom, V_left, V_right, V_top, V_bottom]
            non_zero_velocities = [v for v in velocities if v != 0]
            if non_zero_velocities:
                U = non_zero_velocities[0]
                self.console.insert(tk.END, f'Using velocity {U} to compute viscosity\n')
            else:
                U = U_default
                self.console.insert(tk.END, f'Warning: All input velocities are zero, using default velocity {U_default}\n')

            nu = U * Ly / Re
            dx = Lx / (nx - 1)
            dy = Ly / (ny - 1)
            alpha = nu / Pr

            U_max = max(abs(v) for v in velocities) or U_default
            dt_adv = 0.5 * min(dx, dy) / U_max
            dt_diff = 0.25 * min(dx*dx, dy*dy) / max(nu, alpha)
            dt_safe = min(dt_adv, dt_diff)

        
            self.console.insert(tk.END, f'Reynolds Number = {Re}\n')
            self.console.insert(tk.END, f'Prandtl Number = {Pr}\n')
            self.console.insert(tk.END, f'Current Time Step = {dt}\n')
            self.console.insert(tk.END, f'Safe Time Step <= {dt_safe}\n')
            self.console.insert(tk.END, f'Fluid Viscosity = {nu}\n')
            self.console.insert(tk.END, f'Thermal Diffusivity = {alpha}\n')
            self.console.insert(tk.END, f'Grid Size nx = {nx}, ny = {ny}\n')
            self.console.insert(tk.END, f'Actual Grid Spacing dx = {dx}, dy = {dy}\n')

            omega = np.zeros((ny, nx))
            psi = np.zeros((ny, nx))
            u_full = np.zeros((ny, nx))
            v_full = np.zeros((ny, nx))
            T = np.ones((ny, nx)) * T_bottom

            skip = max(1, nt // 200)
            psi_hist, omega_hist, u_hist, v_hist, T_hist = [], [], [], [], []

            coefdx = dx**2 / (2 * (dx**2 + dy**2))
            coefdy = dy**2 / (2 * (dx**2 + dy**2))
            coefdxdy = dx**2 * dy**2 / (2 * (dx**2 + dy**2))

            def apply_velocity_bc(psi, omega, u_full, v_full):
                y = np.linspace(0, Ly, ny)
                psi[:, 0] = U_left * y
                psi[:, -1] = psi[:, -2]
                psi[0, :] = 0.0
                psi[-1, :] = U_left * Ly
                if velocity_bc_left:
                    u_full[:, 0] = u_full[:, 1]
                    v_full[:, 0] = v_full[:, 1]
                else:
                    u_full[:, 0] = U_left
                    v_full[:, 0] = U_left
                if velocity_bc_right:
                    u_full[:, -1] = u_full[:, -2]
                    v_full[:, -1] = v_full[:, -2]
                else:
                    u_full[:, -1] = U_right
                    v_full[:, -1] = V_right
                if velocity_bc_bottom:
                    u_full[0, :] = u_full[1, :]
                    v_full[0, :] = v_full[1, :]
                else:
                    u_full[0, :] = U_bottom
                    v_full[0, :] = V_bottom
                if velocity_bc_top:
                    u_full[-1, :] = u_full[-2, :]
                    v_full[-1, :] = v_full[-2, :]
                else:
                    u_full[-1, :] = U_top
                    v_full[-1, :] = V_top
                omega[:, 0] = 0.0
                omega[:, -1] = omega[:, -2]
                omega[0, :] = (psi[0, :] - psi[1, :]) * 2.0 / dy**2 - 2.0 * u_full[0, :] / dy
                omega[-1, :] = (psi[-1, :] - psi[-2, :]) * 2.0 / dy**2 - 2.0 * u_full[-1, :] / dy
                return psi, omega, u_full, v_full

            def apply_temperature_bc(T):
                if temperature_bc_top:
                    T[-1, :] = T[-2, :]
                else:
                    T[-1, :] = T_top
                if temperature_bc_bottom:
                    T[0, :] = T[1, :]
                else:
                    T[0, :] = T_bottom
                if temperature_bc_left:
                    T[:, 0] = T[:, 1]
                else:
                    T[:, 0] = T_left
                if temperature_bc_right:
                    T[:, -1] = T[:, -2]
                else:
                    T[:, -1] = T_right
                return T

            for p in range(nt):
                if p > 0:
                    psi, omega, u_full, v_full = apply_velocity_bc(psi, omega, u_full, v_full)
                    px = -(psi[2:,1:-1] - psi[:-2,1:-1]) * (omega[1:-1,2:] - omega[1:-1,:-2]) / (4*dy*dx)
                    py = (omega[2:,1:-1] - omega[:-2,1:-1]) * (psi[1:-1,2:] - psi[1:-1,:-2]) / (4*dy*dx)
                    pxy = (omega[1:-1,2:] + omega[1:-1,:-2] - 2*omega[1:-1,1:-1]) / dx**2 \
                        + (omega[2:,1:-1] + omega[:-2,1:-1] - 2*omega[1:-1,1:-1]) / dy**2
                    omega[1:-1,1:-1] += dt * (px + py + nu * pxy)
                    err = 1.0
                    counter = 0
                    while err > 1e-5 and counter < 200:
                        psi_old = psi.copy()
                        psi[1:-1,1:-1] = coefdxdy * omega[1:-1,1:-1] \
                                      + coefdx * (psi[2:,1:-1] + psi[:-2,1:-1]) \
                                      + coefdy * (psi[1:-1,2:] + psi[1:-1,:-2])
                        err = np.linalg.norm(psi - psi_old, ord=np.inf)
                        counter += 1
                    u_full[1:-1,1:-1] = (psi[2:,1:-1] - psi[:-2,1:-1]) / (2*dy)
                    v_full[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
                    psi, omega, u_full, v_full = apply_velocity_bc(psi, omega, u_full, v_full)
                    T = apply_temperature_bc(T)
                    px = -(psi[2:,1:-1] - psi[:-2,1:-1]) * (T[1:-1,2:] - T[1:-1,:-2]) / (4*dy*dx)
                    py = (T[2:,1:-1] - T[:-2,1:-1]) * (psi[1:-1,2:] - psi[1:-1,:-2]) / (4*dy*dx)
                    pxy = (T[1:-1,2:] + T[1:-1,:-2] - 2*T[1:-1,1:-1]) / dx**2 \
                        + (T[2:,1:-1] + T[:-2,1:-1] - 2*T[1:-1,1:-1]) / dy**2
                    T[1:-1,1:-1] += dt * (px + py + alpha * pxy)
                if p % skip == 0:
                    psi_hist.append(psi.copy())
                    omega_hist.append(omega.copy())
                    u_hist.append(u_full.copy())
                    v_hist.append(v_full.copy())
                    T_hist.append(T.copy())

            x = np.linspace(0, Lx, nx)
            y = np.linspace(0, Ly, ny)
            X, Y = np.meshgrid(x, y)
            y_slice = Ly / 2
            j_slice = int(np.round(y_slice / Ly * (ny - 1)))

            if self.canvas is not None:
                self.canvas.get_tk_widget().destroy()
            if self.fig is not None:
                plt.close(self.fig)
            if self.anim is not None:
                self.anim.event_source.stop()

            self.fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=100)
            self.fig.patch.set_facecolor('#f8f9fa' if self.current_theme == 'litera' else '#343a40')
            axes = axes.flatten()

            plt.subplots_adjust(wspace=0.4, hspace=0.5)

            plots = {
                'mag': {'ax': axes[0], 'contour': None, 'stream': None,
                        'title': 'Velocity Magnitude and Streamlines',
                        'levels': np.arange(0, U_max + 0.1, 0.1),
                        'cbar_label': 'Velocity Magnitude (m/s)'},
                'u': {'ax': axes[1], 'contour': None,
                      'title': 'Horizontal Velocity (u)',
                      'levels': np.arange(-U_max*0.5, U_max + 0.1, 0.1),
                      'cbar_label': 'u-Velocity (m/s)'},
                'v': {'ax': axes[2], 'contour': None,
                      'title': 'Vertical Velocity (v)',
                      'levels': np.arange(-U_max*0.3, U_max*0.3 + 0.01, 0.03),
                      'cbar_label': 'v-Velocity (m/s)'},
                'T': {'ax': axes[3], 'contour': None,
                      'title': 'Temperature Field',
                      'levels': np.linspace(min(T_bottom, T_left, T_right, T_top),
                                           max(T_bottom, T_left, T_right, T_top), 21),
                      'cbar_label': 'Temperature (K)'},
                'line': {'ax': axes[4], 'lines': None,
                         'title': f'y={y_slice:.2f} Velocity Distribution',
                         'xlabel': 'x (m)', 'ylabel': 'Velocity (m/s)'},
                'T_line': {'ax': axes[5], 'lines': None,
                           'title': f'y={y_slice:.2f} Temperature Distribution',
                           'xlabel': 'x (m)', 'ylabel': 'Temperature (K)'}
            }

            for key in plots:
                ax = plots[key]['ax']
                ax.set_title(plots[key]['title'], fontfamily='Arial', fontsize=10, 
                            color='#000000' if self.current_theme == 'litera' else '#ffffff')
                if key not in ['line', 'T_line']:
                    ax.set_aspect('equal')
                    ax.set_xlabel('x (m)', fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.set_ylabel('y (m)', fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.set_xlim(0, Lx)
                    ax.set_ylim(0, Ly)
                    ax.tick_params(colors='#000000' if self.current_theme == 'litera' else '#ffffff')
                    plots[key]['contour'] = ax.contourf(X, Y, np.zeros_like(X),
                                                        levels=plots[key]['levels'],
                                                        cmap=cm.jet, extend='both')
                    cbar = self.fig.colorbar(plots[key]['contour'], ax=ax,
                                        orientation='vertical',
                                        label=plots[key]['cbar_label'])
                    cbar.ax.yaxis.set_tick_params(
                        color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    cbar.ax.yaxis.set_label_text(
                        plots[key]['cbar_label'], 
                        color='#000000' if self.current_theme == 'litera' else '#ffffff')
                else:
                    ax.set_xlabel(plots[key]['xlabel'], fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.set_ylabel(plots[key]['ylabel'], fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.tick_params(colors='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.grid(True, color='#888888')

            def update(frame):
                u = u_hist[frame]
                v = v_hist[frame]
                T = T_hist[frame]
                mag = np.sqrt(u**2 + v**2)
                self.fig.suptitle(f'Time Step: {frame}/{len(u_hist)-1}', y=1.02, fontsize=12, 
                                 fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')

                for key in ['mag', 'u', 'v', 'T']:
                    ax = plots[key]['ax']
                    if plots[key]['contour'] is not None:
                        for artist in ax.collections[:]:
                            if isinstance(artist, matplotlib.contour.QuadContourSet):
                                artist.remove()
                    if key == 'mag':
                        plots[key]['contour'] = ax.contourf(X, Y, mag, levels=plots[key]['levels'], cmap=cm.jet, extend='both')
                    elif key == 'u':
                        plots[key]['contour'] = ax.contourf(X, Y, u, levels=plots[key]['levels'], cmap=cm.jet, extend='both')
                    elif key == 'v':
                        plots[key]['contour'] = ax.contourf(X, Y, v, levels=plots[key]['levels'], cmap=cm.jet, extend='both')
                    elif key == 'T':
                        plots[key]['contour'] = ax.contourf(X, Y, T, levels=plots[key]['levels'], cmap=cm.jet, extend='both')

                if plots['mag']['stream'] is not None:
                    if hasattr(plots['mag']['stream'], 'lines'):
                        plots['mag']['stream'].lines.remove()
                plots['mag']['stream'] = plots['mag']['ax'].streamplot(
                    X, Y, u, v, density=1.0, color='w', linewidth=1.0)

                if plots['line']['lines'] is not None:
                    for line in plots['line']['lines']:
                        line.remove()
                plots['line']['lines'] = []
                plots['line']['lines'].append(plots['line']['ax'].plot(x, u[j_slice, :], 'r-', label='u')[0])
                plots['line']['lines'].append(plots['line']['ax'].plot(x, v[j_slice, :], 'g-', label='v')[0])
                plots['line']['ax'].legend()

                if plots['T_line']['lines'] is not None:
                    for line in plots['T_line']['lines']:
                        line.remove()
                plots['T_line']['lines'] = []
                plots['T_line']['lines'].append(plots['T_line']['ax'].plot(x, T[j_slice, :], 'k-', label='T')[0])
                plots['T_line']['ax'].legend()

                return [plots[key]['contour'] for key in ['mag', 'u', 'v', 'T']] + \
                       [plots['mag']['stream'].lines] + plots['line']['lines'] + plots['T_line']['lines']

            self.plot_label.destroy()
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root.winfo_children()[0].winfo_children()[1])
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=5)
            self.root.winfo_children()[0].winfo_children()[1].columnconfigure(0, weight=1)
            self.root.winfo_children()[0].winfo_children()[1].rowconfigure(0, weight=1)

            frame_step = max(1, len(u_hist) // 40)
            self.anim = animation.FuncAnimation(
                self.fig, update,
                frames=range(0, len(u_hist), frame_step),
                init_func=lambda: update(0),
                interval=200,
                blit=False
            )
            self.canvas.draw()
            self.console.insert(tk.END, "Simulation completed and animation displayed in GUI.\n")

        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {e}")
            self.console.insert(tk.END, f"Error during simulation: {e}\n")

if __name__ == "__main__":
    root = ttk.Window()
    app = CFDSimulationGUI(root)
    root.mainloop()