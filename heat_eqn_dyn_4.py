import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametry siatki
nx, ny = 100, 100
dx = dy = 1.0
alpha = 0.1
dt = 0.1
steps = 200

# Macierz temperatury
u = np.zeros((nx, ny))

# Początkowe źródło ciepła
u[nx//2, ny//2] = 100

# Parametry wentylatora
fan_width = 5
fan_height = 10
fan_y = ny // 2 - fan_height // 2
fan_x = 0
fan_direction = 1  # 1 = w prawo, -1 = w lewo

# Funkcja dodająca ciepło
def add_heat(x, y, radius=3, temp=100):
    for i in range(nx):
        for j in range(ny):
            if (i - x)**2 + (j - y)**2 <= radius**2:
                u[i, j] = temp

# Obsługa kliknięcia myszy
def onclick(event):
    if event.inaxes:
        ix, iy = int(event.xdata), int(event.ydata)
        if 0 <= ix < nx and 0 <= iy < ny:
            if event.button == 1:
                add_heat(ix, iy, radius=3, temp=100)
            elif event.button == 3:
                add_heat(ix, iy, radius=3, temp=-50)

# Aktualizacja animacji
def update(frame):
    global u, fan_x, fan_direction

    u_new = u.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i, j] = u[i, j] + alpha * dt * (
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            )

    # Ruch wentylatora
    fan_x += fan_direction
    if fan_x + fan_width >= nx or fan_x <= 0:
        fan_direction *= -1  # zmiana kierunku

    # Chłodzenie przez wentylator
    u_new[fan_x:fan_x+fan_width, fan_y:fan_y+fan_height] -= 5

    u[:] = u_new
    im.set_array(u)
    return [im]

# Tworzenie wykresu
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='coolwarm', interpolation='nearest', vmin=-100, vmax=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Temperatura [°C]")
ax.invert_yaxis()
fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Kliknij, aby dodać ciepło (LPM) lub chłodzenie (PPM)")

ani = animation.FuncAnimation(fig, update, frames=steps, blit=True)
plt.show()
