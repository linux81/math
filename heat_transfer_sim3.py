import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton


# -------------------------
# Parametry siatki i modelu
# -------------------------
nx, ny = 60, 60
alpha = 0.1
dt = 0.1

# Pole temperatury
u = np.zeros((nx, ny))

# -------------------------
# Stan aplikacji
# -------------------------
sources = []           # lista trwałych źródeł: dict{x,y,r,temp}
is_running = False     # start/pauza
base_interval = 100    # ms, bazowy interwał animacji (1.0x)

# -------------------------
# GUI
# -------------------------
root = tk.Tk()
root.title("Symulacja przewodnictwa ciepła")

# Zmienne GUI (po utworzeniu root!)
source_counter = tk.IntVar(value=0)
status_text = tk.StringVar(value="Oczekiwanie na start")
frame_counter = tk.IntVar(value=0)
speed_factor = tk.DoubleVar(value=1.0)      # mnożnik prędkości (1.0x)
speed_label_text = tk.StringVar(value="1.0x")
current_cmap = tk.StringVar(value='coolwarm')
heat_temp = tk.IntVar(value=100)
cooling_strength = tk.IntVar(value=50)

# -------------------------
# Wykres (matplotlib w Tk)
# -------------------------
fig, ax = plt.subplots()
im = ax.imshow(u, cmap=current_cmap.get(), interpolation='nearest', vmin=-150, vmax=150)
ax.invert_yaxis()
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Temperatura [°C]")

# -------------------------
# Logika źródeł
# -------------------------
def apply_sources(field):
    """Narzuca temperaturę w obszarach wszystkich źródeł."""
    for s in sources:
        x0, y0, r, t = s['x'], s['y'], s['r'], s['temp']
        x_min, x_max = max(0, x0 - r), min(nx - 1, x0 + r)
        y_min, y_max = max(0, y0 - r), min(ny - 1, y0 + r)
        xs = np.arange(x_min, x_max + 1)[:, None]
        ys = np.arange(y_min, y_max + 1)[None, :]
        mask = (xs - x0) ** 2 + (ys - y0) ** 2 <= r ** 2
        field[x_min:x_max + 1, y_min:y_max + 1][mask] = t

def add_source(ix, iy, temp, radius=3):
    sources.append({'x': ix, 'y': iy, 'r': radius, 'temp': temp})
    source_counter.set(len(sources))

def onclick(event):
    if not event.inaxes or event.xdata is None or event.ydata is None:
        return

    ix, iy = int(event.xdata), int(event.ydata)
    if not (0 <= ix < nx and 0 <= iy < ny):
        return

    # Rozpoznanie przycisku: wspiera enum i wartości 1/2/3
    btn = event.button
    key = (event.key or "").lower()  # modyfikatory z klawiatury

    is_left = (btn == MouseButton.LEFT) or (btn == 1)
    is_right = (btn == MouseButton.RIGHT) or (btn == 3)

    # Fallback: LPM + Shift/Ctrl = chłodzenie (gdy PPM nie dociera)
    cooling_via_modifier = is_left and (("shift" in key) or ("control" in key) or ("ctrl" in key))

    if is_right or cooling_via_modifier:
        add_source(ix, iy, -cooling_strength.get(), radius=3)
    elif is_left:
        add_source(ix, iy, heat_temp.get(), radius=3)
    # inne przyciski ignorujemy

fig.canvas.mpl_connect('button_press_event', onclick)

# -------------------------
# Pętla animacji
# -------------------------
def update(frame):
    global u
    # Nakładamy źródła zawsze, żeby były widoczne także w pauzie/przed startem
    u_work = u.copy()
    apply_sources(u_work)

    if is_running:
        u_new = u_work.copy()
        u_new[1:-1, 1:-1] = u_work[1:-1, 1:-1] + alpha * dt * (
            (u_work[2:, 1:-1] - 2 * u_work[1:-1, 1:-1] + u_work[:-2, 1:-1]) +
            (u_work[1:-1, 2:] - 2 * u_work[1:-1, 1:-1] + u_work[1:-1, :-2])
        )
        u[:] = u_new
        frame_counter.set(frame_counter.get() + 1)
    else:
        u[:] = u_work

    im.set_array(u)
    im.set_cmap(current_cmap.get())
    return [im]

# Osadzenie wykresu w Tk
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# -------------------------
# Panel sterowania
# -------------------------
controls = tk.Frame(root)
controls.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

# Mapa kolorów
tk.Label(controls, text="Mapa kolorów:").grid(row=0, column=0, sticky='w', padx=4, pady=2)
ttk.Combobox(controls, textvariable=current_cmap, values=sorted(plt.colormaps()),
             state="readonly", width=16).grid(row=0, column=1, padx=4, pady=2)

# Suwaki
tk.Label(controls, text="Temperatura źródła (LPM):").grid(row=1, column=0, sticky='w', padx=4, pady=2)
tk.Scale(controls, variable=heat_temp, from_=10, to=300, orient=tk.HORIZONTAL, length=220)\
    .grid(row=1, column=1, padx=4, pady=2)

tk.Label(controls, text="Siła chłodzenia (PPM):").grid(row=2, column=0, sticky='w', padx=4, pady=2)
tk.Scale(controls, variable=cooling_strength, from_=10, to=200, orient=tk.HORIZONTAL, length=220)\
    .grid(row=2, column=1, padx=4, pady=2)

# Licznik źródeł
tk.Label(controls, text="Liczba źródeł:").grid(row=3, column=0, sticky='w', padx=4, pady=2)
tk.Label(controls, textvariable=source_counter).grid(row=3, column=1, sticky='w', padx=4, pady=2)

# Status
tk.Label(controls, text="Status:").grid(row=4, column=0, sticky='w', padx=4, pady=2)
status_label = tk.Label(controls, textvariable=status_text, fg="gray")
status_label.grid(row=4, column=1, sticky='w', padx=4, pady=2)

# Czas symulacji (kroki)
tk.Label(controls, text="Czas (kroki):").grid(row=5, column=0, sticky='w', padx=4, pady=2)
tk.Label(controls, textvariable=frame_counter).grid(row=5, column=1, sticky='w', padx=4, pady=2)

# Tempo (mnożnik)
tk.Label(controls, text="Tempo:").grid(row=6, column=0, sticky='w', padx=4, pady=2)
tk.Label(controls, textvariable=speed_label_text).grid(row=6, column=1, sticky='w', padx=4, pady=2)

# -------------------------
# Sterowanie start/pauza/reset
# -------------------------
def start_animation():
    global is_running
    is_running = True
    btn_pause.config(text="Pauza")
    status_text.set("Symulacja działa")
    status_label.config(fg="green")

btn_start = tk.Button(controls, text="Start", command=start_animation, width=10)
btn_start.grid(row=0, column=2, padx=6, pady=2)

def toggle_animation():
    global is_running
    is_running = not is_running
    btn_pause.config(text="Start" if not is_running else "Pauza")
    status_text.set("Symulacja zatrzymana" if not is_running else "Symulacja działa")
    status_label.config(fg="red" if not is_running else "green")

btn_pause = tk.Button(controls, text="Start", command=toggle_animation, width=10)
btn_pause.grid(row=1, column=2, padx=6, pady=2)

def reset_all():
    global u, sources, is_running
    u = np.zeros((nx, ny))
    sources = []
    source_counter.set(0)
    frame_counter.set(0)
    im.set_array(u)
    fig.canvas.draw_idle()
    status_text.set("Oczekiwanie na start")
    status_label.config(fg="gray")
    is_running = False
    btn_pause.config(text="Start")
    # reset prędkości
    speed_factor.set(1.0)
    speed_label_text.set("1.0x")
    ani.event_source.interval = base_interval

tk.Button(controls, text="Reset wszystko", command=reset_all, width=12)\
  .grid(row=2, column=2, padx=6, pady=2)

def reset_simulation_only():
    global u
    u = np.zeros((nx, ny))
    frame_counter.set(0)
    im.set_array(u)
    fig.canvas.draw_idle()
    if is_running:
        status_text.set("Symulacja działa")
        status_label.config(fg="green")
    else:
        status_text.set("Symulacja zatrzymana")
        status_label.config(fg="red")

tk.Button(controls, text="Reset symulacji", command=reset_simulation_only, width=12)\
  .grid(row=3, column=2, padx=6, pady=2)

# -------------------------
# Kontrola prędkości
# -------------------------
def _update_speed_label():
    speed_label_text.set(f"{speed_factor.get():.2f}x")

def increase_speed():
    current = speed_factor.get()
    new = min(current * 2.0, 8.0)
    speed_factor.set(round(new, 2))
    _update_speed_label()
    ani.event_source.interval = int(base_interval / new)

def decrease_speed():
    current = speed_factor.get()
    new = max(current / 2.0, 0.25)
    speed_factor.set(round(new, 2))
    _update_speed_label()
    ani.event_source.interval = int(base_interval / new)

tk.Button(controls, text="⏩ Szybciej", command=increase_speed, width=10)\
  .grid(row=0, column=4, padx=6, pady=2)
tk.Button(controls, text="⏸️ Wolniej", command=decrease_speed, width=10)\
  .grid(row=1, column=4, padx=6, pady=2)

# -------------------------
# Animacja (blit=False dla stabilności z TkAgg)
# -------------------------
ani = animation.FuncAnimation(fig, update, interval=base_interval, blit=False)

# -------------------------
# Zamknięcie
# -------------------------
def on_close():
    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# Inicjalizacja etykiety tempa
_update_speed_label()

# Start GUI
root.mainloop()
