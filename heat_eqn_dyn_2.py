#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 01:37:33 2025

@author: piotrek
"""
#import numpy as np
#import matplotlib.pyplot as plt
#import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametry siatki
nx, ny = 50, 50
dx = dy = 1.0
alpha = 0.1
dt = 0.1
steps = 200

# Macierz temperatury
u = np.zeros((nx, ny))

# Funkcja dodająca źródło ciepła
def add_heat(x, y, radius=3, temp=100):
    for i in range(nx):
        for j in range(ny):
            if (i - x)**2 + (j - y)**2 <= radius**2:
                u[i, j] = temp
                


# Obsługa kliknięcia myszy
def onclick(event):
    if event.inaxes:
        ix, iy = int(event.xdata), int(event.ydata)
        add_heat(ix, iy)
        print(f"Dodano źródło ciepła w ({ix}, {iy})")

# Aktualizacja animacji
def update(frame):
    global u
    u_new = u.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i, j] = u[i, j] + alpha * dt * (
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            )
    # Chłodzenie w wybranym obszarze
    u_new[10:15, 10:15] -= 5  # odejmujemy temperatur
    u[:] = u_new
    im.set_array(u)
    return [im]

# Tworzenie wykresu
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='hot', interpolation='nearest', vmin=0, vmax=100)
fig.canvas.mpl_connect('button_press_event', onclick)

ani = animation.FuncAnimation(fig, update, frames=steps, blit=True)
plt.title("Kliknij, aby dodać źródło ciepła")
plt.show()
