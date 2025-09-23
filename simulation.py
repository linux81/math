import numpy as np


class SimulationState:
    def __init__(self, nx=60, ny=60):
        self.nx, self.ny = nx, ny
        self.u = np.zeros((nx, ny))
        self.sources = []
        self.is_running = False
        self.frame_counter = 0
        self.speed_factor = 1.0
        self.heat_temp = 100
        self.cooling_strength = 50
        self.current_cmap = 'hot'
        self.boundary_types = {
            "top": "Neumann",
            "bottom": "Neumann",
            "left": "Neumann",
            "right": "Neumann"
        }

    def apply_sources(self):
        for s in self.sources:
            x0, y0, r, t = s['x'], s['y'], s['r'], s['temp']
            x_min, x_max = max(0, x0 - r), min(self.nx - 1, x0 + r)
            y_min, y_max = max(0, y0 - r), min(self.ny - 1, y0 + r)
            xs = np.arange(x_min, x_max + 1)[:, None]
            ys = np.arange(y_min, y_max + 1)[None, :]
            mask = (xs - x0) ** 2 + (ys - y0) ** 2 <= r ** 2
            self.u[x_min:x_max + 1, y_min:y_max + 1][mask] = t

    def update(self, alpha=0.1, dt=0.1):
        if not self.is_running:
            self.apply_sources()
            return

        u_new = self.u.copy()
        u_new[1:-1, 1:-1] = self.u[1:-1, 1:-1] + alpha * dt * (
            (self.u[2:, 1:-1] - 2 * self.u[1:-1, 1:-1] + self.u[:-2, 1:-1]) +
            (self.u[1:-1, 2:] - 2 * self.u[1:-1, 1:-1] + self.u[1:-1, :-2])
        )
        self.u[:] = u_new
        self.apply_sources()  # ← teraz źródła są nakładane po kroku
        self.frame_counter += 1
        
        bt = self.boundary_types

        if bt["top"] == "Neumann":
            self.u[:, 0] = self.u[:, 1]
        elif bt["top"] == "Dirichlet":
            self.u[:, 0] = 0
        elif bt["top"] == "Periodic":
            self.u[:, 0] = self.u[:, -2]
        if bt["bottom"] == "Neumann":
            self.u[:, 0] = self.u[:, 1]
        elif bt["bottom"] == "Dirichlet":
            self.u[:, 0] = 0
        elif bt["bottom"] == "Periodic":
            self.u[:, 0] = self.u[:, -2]
            
        if bt["left"] == "Neumann":
            self.u[:, 0] = self.u[:, 1]
        elif bt["left"] == "Dirichlet":
            self.u[:, 0] = 0
        elif bt["left"] == "Periodic":
            self.u[:, 0] = self.u[:, -2]
            
        if bt["right"] == "Neumann":
            self.u[:, 0] = self.u[:, 1]
        elif bt["right"] == "Dirichlet":
            self.u[:, 0] = 0
        elif bt["right"] == "Periodic":
            self.u[:, 0] = self.u[:, -2]


# Analogicznie dla bottom, left, right...





    def add_source(self, x, y, temp, radius=3):
        self.sources.append({'x': x, 'y': y, 'r': radius, 'temp': temp})

    def reset_all(self):
        self.u[:] = 0
        self.sources.clear()
        self.frame_counter = 0
        self.is_running = False
        self.speed_factor = 1.0

    def reset_simulation_only(self):
        self.u[:] = 0
        self.frame_counter = 0
    def resize_grid(self, new_nx, new_ny):
        self.nx = new_nx
        self.ny = new_ny
        self.u = np.zeros((new_nx, new_ny))
        self.sources.clear()
        self.frame_counter = 0

import matplotlib.pyplot as plt       
def get_available_cmaps():
    return sorted(plt.colormaps())

