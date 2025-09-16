import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametry siatki
nx, ny = 200, 200
dx = dy = 1.0
c = 1.0  # prędkość światła (jednostkowa)
dt = 0.5 * dx / c  # warunek Couranta

# Pola
Ez = np.zeros((nx, ny))  # pole elektryczne (z)
Hx = np.zeros((nx, ny))  # pole magnetyczne (x)
Hy = np.zeros((nx, ny))  # pole magnetyczne (y)

# Źródło fali
def source(t):
    return np.exp(-((t - 30) / 10) ** 2)

# Animacja
fig, ax = plt.subplots()
im = ax.imshow(Ez, cmap='RdBu', vmin=-0.1, vmax=0.1)

def update(frame):
    global Ez, Hx, Hy

    # Aktualizacja pól magnetycznych
    Hx[:, :-1] -= dt / dy * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] += dt / dx * (Ez[1:, :] - Ez[:-1, :])

    # Aktualizacja pola elektrycznego
    Ez[1:, 1:] += dt * (
        (Hy[1:, 1:] - Hy[:-1, 1:]) / dx -
        (Hx[1:, 1:] - Hx[1:, :-1]) / dy
    )

    # Dodanie źródła w centrum
    Ez[nx//2, ny//2] += source(frame)

    im.set_array(Ez)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.title("Symulacja fali elektromagnetycznej 2D (FDTD)")
plt.show()
