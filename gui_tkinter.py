import tkinter as tk
from tkinter import ttk
from simulation import get_available_cmaps

def run_tkinter(state):
    print("Tkinter window starting...")
    root = tk.Tk()
    root.title("Sterowanie symulacją")
    
    def update_boundary_edge(edge):
        def inner(event):
            state.boundary_types[edge] = edge_vars[edge].get()
        return inner
    
    edge_vars = {}
    for i, edge in enumerate(["top", "bottom", "left", "right"]):
        edge_vars[edge] = tk.StringVar(value="Neumann")
        ttk.Label(root, text=f"{edge.capitalize()} brzeg:").grid(row=8+i, column=0)
        box = ttk.Combobox(root, textvariable=edge_vars[edge],
                           values=["Neumann", "Dirichlet", "Periodic"],
                           state="readonly")
        box.grid(row=8+i, column=1)
        box.bind("<<ComboboxSelected>>", update_boundary_edge(edge))


    status = tk.StringVar(value="Zatrzymana")
    speed_label = tk.StringVar(value="1.0x")
    frame_counter_var = tk.StringVar(value="0")
    heat_value = tk.StringVar(value=str(state.heat_temp))
    cool_value = tk.StringVar(value=str(state.cooling_strength))


    def toggle():
        state.is_running = not state.is_running
        status.set("Działa" if state.is_running else "Zatrzymana")

    def reset_all():
        state.reset_all()
        status.set("Zatrzymana")
        speed_label.set("1.0x")

    def reset_sim():
        state.reset_simulation_only()

    def update_heat(val):
        state.heat_temp = int(float(val))
        heat_value.set(f"{state.heat_temp}°C")

    def update_cool(val):
        state.cooling_strength = int(float(val))
        cool_value.set(f"{state.cooling_strength}°C")

    def update_speed_label():
        speed_label.set(f"{state.speed_factor:.2f}x")

    def increase_speed():
        state.speed_factor = min(state.speed_factor * 2.0, 8.0)
        update_speed_label()

    def decrease_speed():
        state.speed_factor = max(state.speed_factor / 2.0, 0.25)
        update_speed_label()

    def update_cmap(event):
        state.current_cmap = cmap_box.get()

    def periodic_update():
        frame_counter_var.set(str(state.frame_counter))
        root.after(500, periodic_update)

    # GUI layout
    ttk.Label(root, text="Status:").grid(row=0, column=0)
    ttk.Label(root, textvariable=status).grid(row=0, column=1)

    ttk.Button(root, text="Start/Pauza", command=toggle).grid(row=1, column=0)
    ttk.Button(root, text="Reset wszystko", command=reset_all).grid(row=1, column=1)
    ttk.Button(root, text="Reset symulacji", command=reset_sim).grid(row=1, column=2)

    ttk.Label(root, text="Temperatura źródła (LPM):").grid(row=2, column=0)
    heat_slider = ttk.Scale(root, from_=10, to=300, orient='horizontal', command=update_heat)
    heat_slider.set(state.heat_temp)
    heat_slider.grid(row=2, column=1)

    ttk.Label(root, text="Siła chłodzenia (PPM):").grid(row=3, column=0)
    cool_slider = ttk.Scale(root, from_=10, to=200, orient='horizontal', command=update_cool)
    cool_slider.set(state.cooling_strength)
    cool_slider.grid(row=3, column=1)

    ttk.Label(root, text="Tempo:").grid(row=4, column=0)
    ttk.Label(root, textvariable=speed_label).grid(row=4, column=1)
    ttk.Button(root, text="⏩ Szybciej", command=increase_speed).grid(row=4, column=2)
    ttk.Button(root, text="⏸️ Wolniej", command=decrease_speed).grid(row=4, column=3)

    ttk.Label(root, text="Mapa kolorów:").grid(row=5, column=0)
    cmap_box = ttk.Combobox(root, values=get_available_cmaps(), state="readonly")
    cmap_box.set(state.current_cmap)
    cmap_box.grid(row=5, column=1)
    cmap_box.bind("<<ComboboxSelected>>", update_cmap)

    ttk.Label(root, text="Czas (kroki):").grid(row=6, column=0)
    ttk.Label(root, textvariable=frame_counter_var).grid(row=6, column=1)
    
    ttk.Label(root, text="Temperatura źródła (LPM):").grid(row=2, column=0)
    heat_slider = ttk.Scale(root, from_=10, to=300, orient='horizontal', command=update_heat)
    heat_slider.set(state.heat_temp)
    heat_slider.grid(row=2, column=1)
    ttk.Label(root, textvariable=heat_value).grid(row=2, column=2)

    ttk.Label(root, text="Siła chłodzenia (PPM):").grid(row=3, column=0)
    cool_slider = ttk.Scale(root, from_=10, to=200, orient='horizontal', command=update_cool)
    cool_slider.set(state.cooling_strength)
    cool_slider.grid(row=3, column=1)
    ttk.Label(root, textvariable=cool_value).grid(row=3, column=2)
    
    ttk.Label(root, text="Zakres LPM:").grid(row=2, column=3)
    heat_min = tk.Entry(root, width=5)
    heat_min.insert(0, "10")
    heat_min.grid(row=2, column=4)
    heat_max = tk.Entry(root, width=5)
    heat_max.insert(0, "300")
    heat_max.grid(row=2, column=5)

    def update_heat_range():
        min_val = int(heat_min.get())
        max_val = int(heat_max.get())
        heat_slider.config(from_=min_val, to=max_val)

    ttk.Button(root, text="Zmień zakres", command=update_heat_range).grid(row=2, column=6)
    
    
    ttk.Label(root, text="Zakres PPM:").grid(row=3, column=3)
    cool_min = tk.Entry(root, width=5)
    cool_min.insert(0, "10")
    cool_min.grid(row=3, column=4)
    cool_max = tk.Entry(root, width=5)
    cool_max.insert(0, "200")
    cool_max.grid(row=3, column=5)

    def update_cool_range():
        min_val = int(cool_min.get())
        max_val = int(cool_max.get())
        cool_slider.config(from_=min_val, to=max_val)

    ttk.Button(root, text="Zmień zakres", command=update_cool_range).grid(row=3, column=6)
    
    
    ttk.Label(root, text="Rozmiar X:").grid(row=16, column=0)
    nx_entry = ttk.Entry(root, width=6)
    nx_entry.insert(0, str(state.nx))
    nx_entry.grid(row=16, column=1)

    ttk.Label(root, text="Rozmiar Y:").grid(row=17, column=0)
    ny_entry = ttk.Entry(root, width=6)
    ny_entry.insert(0, str(state.ny))
    ny_entry.grid(row=17, column=1)
    
    
    grid_size_var = tk.StringVar(value=f"{state.nx} × {state.ny}")

    ttk.Label(root, text="Rozmiar siatki:").grid(row=19, column=0)
    ttk.Label(root, textvariable=grid_size_var).grid(row=19, column=1)


    def apply_resize():
        try:
            new_nx = int(nx_entry.get())
            new_ny = int(ny_entry.get())
            state.resize_grid(new_nx, new_ny)
            grid_size_var.set(f"{new_nx} × {new_ny}")
        except ValueError:
            print("Nieprawidłowe wartości rozmiaru")

    ttk.Button(root, text="Zmień rozmiar siatki", command=apply_resize).grid(row=18, column=0, columnspan=2)




    




    root.after(500, periodic_update)
    root.mainloop()

