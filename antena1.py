import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- parametry siatki i czasu ---
nx, ny = 220, 220
dx = dy = 1.0
c0 = 1.0
eps0 = 1.0
mu0  = 1.0
S = 0.5
dt = S * min(dx, dy) / c0

# --- PML ---
pml = 20
sigma_max = 2.0
order = 3

def profile(idx, N, order):
    x = (idx / N)**order
    return x

sigma_x = np.zeros((nx, ny))
sigma_y = np.zeros((nx, ny))

for i in range(pml):
    s = sigma_max * profile(pml - i, pml, order)
    sigma_x[i, :] += s
    sigma_x[nx-1-i, :] += s
for j in range(pml):
    s = sigma_max * profile(pml - j, pml, order)
    sigma_y[:, j] += s
    sigma_y[:, ny-1-j] += s

# --- współczynniki krokowe (TEz) ---
eps_r = np.ones((nx, ny))

Ceze  = (1 - dt * (sigma_x + sigma_y) / (2 * (eps0 * eps_r))) / (1 + dt * (sigma_x + sigma_y) / (2 * (eps0 * eps_r)))
Cezhx = (dt / (eps0 * eps_r * dx)) / (1 + dt * (sigma_x + sigma_y) / (2 * (eps0 * eps_r)))
Cezhy = (dt / (eps0 * eps_r * dy)) / (1 + dt * (sigma_x + sigma_y) / (2 * (eps0 * eps_r)))

Chxh  = (1 - dt * sigma_y / (2 * mu0)) / (1 + dt * sigma_y / (2 * mu0))
Chxez = (dt / (mu0 * dy)) / (1 + dt * sigma_y / (2 * mu0))
Chyh  = (1 - dt * sigma_x / (2 * mu0)) / (1 + dt * sigma_x / (2 * mu0))
Chyez = (dt / (mu0 * dx)) / (1 + dt * sigma_x / (2 * mu0))

# --- pola ---
Ez = np.zeros((nx, ny))
Hx = np.zeros((nx, ny))
Hy = np.zeros((nx, ny))

# --- źródło antenowe: krótka linia w centrum ---
cx, cy = nx//2, ny//2
half_len = 6  # długość odcinka źródła (2*half_len+1 komórek)
f0 = 0.025    # „częstotliwość” w jednostkach siatkowych
t0 = 80
tau = 25

def src_t(t):
    env = np.exp(-((t - t0)/tau)**2)
    return env * np.sin(2*np.pi*f0*t)

src_indices = [(cx, cy + k) for k in range(-half_len, half_len+1)]

# --- wizualizacja ---
fig, ax = plt.subplots()
im = ax.imshow(Ez.T, cmap='RdBu', vmin=-0.06, vmax=0.06, origin='lower', interpolation='bilinear')
ax.set_title('Antena liniowa 2D z PML (TEz)')

def update(frame):
    global Ez, Hx, Hy

    dE_dy = np.zeros_like(Ez); dE_dy[:, :-1] = Ez[:, 1:] - Ez[:, :-1]
    Hx = Chxh * Hx - Chxez * dE_dy

    dE_dx = np.zeros_like(Ez); dE_dx[:-1, :] = Ez[1:, :] - Ez[:-1, :]
    Hy = Chyh * Hy + Chyez * dE_dx

    dHy_dx = np.zeros_like(Ez); dHy_dx[1:, :] = Hy[1:, :] - Hy[:-1, :]
    dHx_dy = np.zeros_like(Ez); dHx_dy[:, 1:] = Hx[:, 1:] - Hx[:, :-1]
    Ez = Ceze * Ez + Cezhy * dHy_dx - Cezhx * dHx_dy

    s = src_t(frame)
    for (i, j) in src_indices:
        Ez[i, j] += s

    im.set_array(Ez.T)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=600, interval=25, blit=True)
plt.show()
