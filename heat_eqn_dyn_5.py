import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Parametry siatki
nx, ny = 60, 60
alpha = 0.1
dt = 0.1
u = np.zeros((nx, ny))
u[nx//2, ny//2] = 100  # początkowe źródło ciepła

# Lista dostępnych map kolorów
available_cmaps = sorted(plt.colormaps())

# Tworzenie okna Tkinter
root = tk.Tk()
root.title("Symulacja ciepła z wyborem cmap")

# Zmienna cmap
current_cmap = tk.StringVar(value='coolwarm')

# Tworzenie figury matplotlib
fig, ax = plt.subplots()
im = ax.imshow(u, cmap=current_cmap.get(), interpolation='nearest', vmin=-100, vmax=100)
ax.invert_yaxis()
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Temperatura [°C]")

# Obsługa kliknięcia myszy
def onclick(event):
    if event.inaxes:
        ix, iy = int(event.xdata), int(event.ydata)
        if 0 <= ix < nx and 0 <= iy < ny:
            radius = 3
            temp = 100 if event.button == 1 else -50
            for i in range(nx):
                for j in range(ny):
                    if (i - ix)**2 + (j - iy)**2 <= radius**2:
                        u[i, j] = temp

fig.canvas.mpl_connect('button_press_event', onclick)

# Aktualizacja animacji
def update(frame):
    global u
    u_new = u.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i, j] = u[i, j] + alpha * dt * (
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) +
                (u[i, j+1] - 2*u[i, j] + u[i, j-1])
            )
    u[:] = u_new
    im.set_array(u)
    im.set_cmap(current_cmap.get())
    return [im]

# Osadzenie wykresu w oknie Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Interfejs wyboru cmap
frame_controls = tk.Frame(root)
frame_controls.pack(side=tk.BOTTOM, fill=tk.X)

label = tk.Label(frame_controls, text="Wybierz mapę kolorów:")
label.pack(side=tk.LEFT, padx=10)

cmap_menu = ttk.Combobox(frame_controls, textvariable=current_cmap, values=available_cmaps, state="readonly")
cmap_menu.pack(side=tk.LEFT, padx=10)

# Animacja
ani = animation.FuncAnimation(fig, update, interval=100, blit=True)

# Obsługa zamknięcia
def on_close():
    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
