import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametry siatki
nx, ny = 200, 200
dx = dy = 1.0
c0 = 1.0         # jednostkowa prędkość światła
eps0 = 1.0
mu0 = 1.0
S = 0.5          # liczba Couranta
dt = S * min(dx, dy) / c0

# PML
pml_size = 20
sigma_max = 2.0   # moc tłumienia PML (zwiększ, by mieć mocniejszą absorpcję)
order = 3

def grade_profile(n, N, order):
    x = (n / N)**order
    return x

sigma_x = np.zeros((nx, ny))
sigma_y = np.zeros((nx, ny))

# Strefy PML przy brzegach x
for i in range(pml_size):
    s = sigma_max * grade_profile(pml_size - i, pml_size, order)
    sigma_x[i, :] += s
    sigma_x[nx - 1 - i, :] += s

# Strefy PML przy brzegach y
for j in range(pml_size):
    s = sigma_max * grade_profile(pml_size - j, pml_size, order)
    sigma_y[:, j] += s
    sigma_y[:, ny - 1 - j] += s

# Współczynniki UPML (TEz: split-field)
# Wzory typu: d/dt(Hx) = -1/mu (∂Ez/∂y) - (sigma_y/mu) Hx  itd.
# Implementujemy to jako współczynniki krokowe.
Ceze = (1 - dt * (sigma_x + sigma_y) / (2 * eps0)) / (1 + dt * (sigma_x + sigma_y) / (2 * eps0))
Cezhx = (dt / (eps0 * dx)) / (1 + dt * (sigma_x + sigma_y) / (2 * eps0))
Cezhy = (dt / (eps0 * dy)) / (1 + dt * (sigma_x + sigma_y) / (2 * eps0))

Chxh = (1 - dt * sigma_y / (2 * mu0)) / (1 + dt * sigma_y / (2 * mu0))
Chxez = (dt / (mu0 * dy)) / (1 + dt * sigma_y / (2 * mu0))

Chyh = (1 - dt * sigma_x / (2 * mu0)) / (1 + dt * sigma_x / (2 * mu0))
Chyez = (dt / (mu0 * dx)) / (1 + dt * sigma_x / (2 * mu0))

# Pola
Ez  = np.zeros((nx, ny))
Hx  = np.zeros((nx, ny))
Hy  = np.zeros((nx, ny))

# Źródło: impuls Gaussa w czasie, dipol w środku
def src_t(t):
    return np.exp(-((t - 30) / 10.0) ** 2)

src_i, src_j = nx // 2, ny // 2

# Przeszkoda: dielektryk (opcjonalnie komentuj/odkomentuj)
eps_r = np.ones((nx, ny))
# eps_r[nx//2-10:nx//2+10, ny//2+20:ny//2+25] = 4.0  # prosty falowód/przeszkoda
# Zmodyfikuj współczynniki na podstawie eps_r
Ceze = (1 - dt * (sigma_x + sigma_y) / (2 * (eps0 * eps_r))) / (1 + dt * (sigma_x + sigma_y) / (2 * (eps0 * eps_r)))
Cezhx = (dt / (eps0 * eps_r * dx)) / (1 + dt * (sigma_x + sigma_y) / (2 * (eps0 * eps_r)))
Cezhy = (dt / (eps0 * eps_r * dy)) / (1 + dt * (sigma_x + sigma_y) / (2 * (eps0 * eps_r)))

# Animacja
fig, ax = plt.subplots()
im = ax.imshow(Ez.T, cmap='RdBu', vmin=-0.08, vmax=0.08, origin='lower', interpolation='bilinear')
ax.set_title('FDTD 2D TEz z PML')

def update(frame):
    global Ez, Hx, Hy

    # Hx (różnice po y)
    dE_dy = np.zeros_like(Ez)
    dE_dy[:, :-1] = Ez[:, 1:] - Ez[:, :-1]
    Hx = Chxh * Hx - Chxez * dE_dy

    # Hy (różnice po x)
    dE_dx = np.zeros_like(Ez)
    dE_dx[:-1, :] = Ez[1:, :] - Ez[:-1, :]
    Hy = Chyh * Hy + Chyez * dE_dx

    # Ez (rotacja H)
    dHy_dx = np.zeros_like(Ez)
    dHy_dx[1:, :] = Hy[1:, :] - Hy[:-1, :]
    dHx_dy = np.zeros_like(Ez)
    dHx_dy[:, 1:] = Hx[:, 1:] - Hx[:, :-1]

    Ez = Ceze * Ez + Cezhy * dHy_dx - Cezhx * dHx_dy

    # Źródło prądowe Jz
    Ez[src_i, src_j] += src_t(frame)

    im.set_array(Ez.T)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=300, interval=30, blit=True)
plt.show()
