import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Jednostki zredukowane
hbar = 1.0
m = 1.0

# Siatka
Nx = 2048
L = 400.0
dx = L / Nx
x = np.linspace(-L/2, L/2 - dx, Nx)

# Przestrzeń p
dk = 2*np.pi / L
k = np.fft.fftfreq(Nx, d=dx) * 2*np.pi

# Potencjał: bariera prostokątna
V0 = 0.015
w = 10.0
V = np.zeros_like(x)
V[(x > 20) & (x < 20 + w)] = V0

# Pakiet Gaussa
x0 = -120.0
p0 = 0.2  # pęd średni (kierunek +x)
sigma = 10.0
psi = (1/(np.pi*sigma**2)**0.25) * np.exp(-(x - x0)**2/(2*sigma**2)) * np.exp(1j*p0*x)

# Czas
dt = 0.5
steps = 1500

# Prefaktory split-step
expV = np.exp(-1j * V * dt / (2*hbar))
expK = np.exp(-1j * (hbar * k**2) * dt / (2*m))

# Przygotowanie wykresu
fig, ax = plt.subplots()
line_psi, = ax.plot(x, np.abs(psi)**2, 'b', lw=1.5, label='|psi|^2')
line_V, = ax.plot(x, V/np.max(V+1e-12)*np.max(np.abs(psi)**2), 'k--', lw=1, label='V (skalowane)')
ax.set_xlim(-150, 150)
ax.set_ylim(0, 0.12)
ax.set_title('1D Schrödinger: rozpraszanie pakietu na barierze')
ax.legend(loc='upper right')

def step():
    global psi
    # pół kroku w potencjale
    psi *= expV
    # pełny krok w kinetyce (FFT)
    psi_k = np.fft.fft(psi)
    psi_k *= expK
    psi = np.fft.ifft(psi_k)
    # pół kroku w potencjale
    psi *= expV

def update(i):
    for _ in range(2):  # 2 kroki na klatkę, żeby przyspieszyć
        step()
    line_psi.set_ydata(np.abs(psi)**2)
    return [line_psi]

ani = animation.FuncAnimation(fig, update, frames=600, interval=20, blit=True)
plt.show()
