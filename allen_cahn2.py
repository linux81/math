import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parametry siatki ---
Nx, Ny = 100, 100
dx = 1.0 / Nx
dy = 1.0 / Ny
dt = 0.01
Nt = 300
epsilon = 0.01

# --- Siatka przestrzenna ---
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)

# --- Pocz?tkowy rozk?ad faz: losowy ---
u = np.random.rand(Ny, Nx) * 2 - 1  # warto?ci w zakresie [-1, 1]

# --- Przygotowanie wykresu ---
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='bwr', origin='lower', extent=[0, 1, 0, 1], vmin=-1, vmax=1)
ax.set_title("R體nanie Allen-Cahna 2D")

# --- Funkcja aktualizuj?ca animacj? ---
def update(frame):
    global u
    u_new = u.copy()
    # Laplacjan (dyfuzja)
    laplacian = (
        u[2:, 1:-1] + u[:-2, 1:-1] +
        u[1:-1, 2:] + u[1:-1, :-2] -
        4 * u[1:-1, 1:-1]
    ) / dx**2
    # Nieliniowy cz?on reakcji
    reaction = u[1:-1, 1:-1]**3 - u[1:-1, 1:-1]
    # Aktualizacja
    u_new[1:-1, 1:-1] += dt * (epsilon**2 * laplacian - reaction)
    u = u_new
    im.set_array(u)
    return [im]

# --- Tworzenie animacji i zapis do GIF ---
ani = animation.FuncAnimation(fig, update, frames=Nt, interval=50)
ani.save("allen_cahn_2d.gif", writer='pillow', fps=20)
plt.show()