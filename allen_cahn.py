import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from datetime import datetime

# --- Parametry symulacji ---
Nx = 256  # Liczba punktów w wymiarze x
Ny = 256  # Liczba punktów w wymiarze y
T = 5000  # Liczba kroków czasowych
dt = 0.1   # Krok czasowy (delta t)
dx = 1.0  # Krok przestrzenny (delta x)
dy = 1.0  # Krok przestrzenny (delta y)
M = 1.0   # Stała M (mobilność)
alpha = 1.0  # Współczynnik gradientu (gamma)
A = 1.0  # Wysokość potencjału podwójnej studni

# --- Inicjalizacja ---
phi = np.random.rand(Ny, Nx)
history = [phi.copy()]

# --- Pętla czasowa ---
for t_step in range(T):
    laplacian_phi = (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) - 2 * phi) / (dy**2) + \
                    (np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 2 * phi) / (dx**2)
    V_prime = A * 2 * phi * (1 - phi) * (1 - 2 * phi)
    d_phi = M * (alpha * laplacian_phi - V_prime)
    phi = phi + dt * d_phi
    if t_step % 50 == 0:
        history.append(phi.copy())

# --- Wizualizacja ---
fig, ax = plt.subplots(figsize=(6, 6))

# Zdefiniowanie nazwy mapy kolorów
cmap_name = 'viridis'
im = ax.imshow(history[0], cmap=cmap_name, vmin=0, vmax=1)
ax.set_title("Symulacja równania Allena-Cahna z losowymi warunkami początkowymi")
ax.set_xlabel("Pozycja (x)")
ax.set_ylabel("Pozycja (y)")

def animate(i):
    """Funkcja do aktualizacji animacji w każdej klatce."""
    im.set_array(history[i])
    return im,

# Tworzenie animacji z historycznych danych
ani = FuncAnimation(fig, animate, frames=len(history), interval=20, blit=True)
plt.show()

# --- Zapisywanie animacji (opcjonalne) ---

# Pobranie aktualnej daty i czasu
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_filename = f"allen_cahn_2d_{timestamp}_{cmap_name}"

# Zapis do pliku MP4
print(f"Zapisuję animację do pliku '{base_filename}.mp4'...")
writer_mp4 = FFMpegWriter(fps=30)
ani.save(f"{base_filename}.mp4", writer=writer_mp4)
print("Zakończono zapis do MP4.")

# Zapis do pliku GIF
print(f"Zapisuję animację do pliku '{base_filename}.gif'...")
writer_gif = PillowWriter(fps=30)
ani.save(f"{base_filename}.gif", writer=writer_gif)
print("Zakończono zapis do GIF.")
