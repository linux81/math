import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime

class SimState:
    def __init__(self):
        self.Lx, self.Ly = 8.0, 4.0
        self.nx, self.ny = 401, 201
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.c = 1.0
        self.gamma = 0.01
        self.t = 0.0
        self.running = True

        self.f1, self.f2 = 0.65, 0.85
        self.A1, self.A2 = 1.0, 1.0
        self.phi1, self.phi2 = 0.0, np.pi/4
        self.d = 1.2
        self.bc = 'neumann'

        self.u_nm1 = np.zeros((self.ny, self.nx))
        self.u_n   = np.zeros((self.ny, self.nx))
        self.u_np1 = np.zeros((self.ny, self.nx))

        sponge_width = int(0.08 * self.nx)
        self.sponge = np.ones((self.ny, self.nx))
        if sponge_width > 0:
            left = np.linspace(1.0, 0.0, sponge_width)**2
            right = left[::-1]
            self.sponge[:, :sponge_width] *= left
            self.sponge[:, -sponge_width:] *= right

        self.probe_x = self.Lx / 2
        self.probe_y = self.Ly / 2
        self.ix_probe, self.iy_probe = self.nearest_idx(self.probe_x, self.probe_y)

        self.fft_buffer_len = 2048
        self.probe_buffer = np.zeros(self.fft_buffer_len)
        self.probe_ptr = 0
        self.probe_times = []

        self.cmap = 'RdBu_r'
        self.recording = False
        self.record_frames = []
        self.record_format = 'mp4'
        self.saving_status = None
        self.saving_progress = 0

        self.lock = threading.Lock()

    def compute_dt(self):
        return 0.9 / (self.c * np.sqrt((1/self.dx**2) + (1/self.dy**2)))

    def source_positions(self):
        x1 = self.Lx/2 - self.d/2
        x2 = self.Lx/2 + self.d/2
        y0 = self.Ly/2
        return (x1, y0), (x2, y0)

    def nearest_idx(self, x0, y0):
        ix = int(np.clip(np.round((x0 - self.x[0]) / self.dx), 0, self.nx-1))
        iy = int(np.clip(np.round((y0 - self.y[0]) / self.dy), 0, self.ny-1))
        return ix, iy

def apply_vertical_bc(state, U):
    if state.bc == 'neumann':
        U[0, :]  = U[1, :]
        U[-1, :] = U[-2, :]
    else:
        U[0, :]  = 0.0
        U[-1, :] = 0.0

def add_sources(state, U, dt_local, t_now):
    (sx1, sy1), (sx2, sy2) = state.source_positions()
    ix1, iy1 = state.nearest_idx(sx1, sy1)
    ix2, iy2 = state.nearest_idx(sx2, sy2)
    s1 = state.A1 * np.sin(2*np.pi*state.f1 * t_now + state.phi1)
    s2 = state.A2 * np.sin(2*np.pi*state.f2 * t_now + state.phi2)
    U[iy1, ix1] += (dt_local**2) * s1
    U[iy2, ix2] += (dt_local**2) * s2

def launch_animation_window(state: SimState):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(state.u_n, extent=[0, state.Lx, 0, state.Ly], origin='lower',
                   cmap=state.cmap, vmin=-1.5, vmax=1.5, interpolation='bilinear')
    ax.set_title('Pole falowe u(x, y, t)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    (sx1, sy1), (sx2, sy2) = state.source_positions()
    s1 = ax.scatter([sx1], [sy1], c='k', marker='x', s=60)
    s2 = ax.scatter([sx2], [sy2], c='k', marker='x', s=60)
    probe_marker = ax.scatter([state.probe_x], [state.probe_y], c='lime', marker='o', s=60)

    def on_click(event):
        if event.inaxes != ax:
            return
        with state.lock:
            state.probe_x = float(np.clip(event.xdata, 0.0, state.Lx))
            state.probe_y = float(np.clip(event.ydata, 0.0, state.Ly))
            state.ix_probe, state.iy_probe = state.nearest_idx(state.probe_x, state.probe_y)
            state.probe_buffer.fill(0.0)
            state.probe_ptr = 0
            state.probe_times.clear()
    fig.canvas.mpl_connect('button_press_event', on_click)

    def update(_frame):
        with state.lock:
            if not state.running:
                return (im,)
            dt = state.compute_dt()
            lap = (
                (np.roll(state.u_n, +1, axis=1) - 2*state.u_n + np.roll(state.u_n, -1, axis=1)) / state.dx**2 +
                (np.roll(state.u_n, +1, axis=0) - 2*state.u_n + np.roll(state.u_n, -1, axis=0)) / state.dy**2
            )
            state.u_np1 = (2.0 - state.gamma*dt) * state.u_n - (1.0 - state.gamma*dt) * state.u_nm1 + (state.c*dt)**2 * lap
            apply_vertical_bc(state, state.u_np1)
            add_sources(state, state.u_np1, dt, state.t)
            state.u_np1 *= state.sponge
            state.u_nm1, state.u_n = state.u_n, state.u_np1

            val = state.u_n[state.iy_probe, state.ix_probe]
            state.probe_buffer[state.probe_ptr % state.fft_buffer_len] = val
            state.probe_ptr += 1
            state.probe_times.append(state.t)
            if len(state.probe_times) > state.fft_buffer_len:
                state.probe_times = state.probe_times[-state.fft_buffer_len:]

            (sx1, sy1), (sx2, sy2) = state.source_positions()
            s1.set_offsets([[sx1, sy1]])
            s2.set_offsets([[sx2, sy2]])
            probe_marker.set_offsets([[state.probe_x, state.probe_y]])
            im.set_cmap(state.cmap)

            vmax = np.percentile(np.abs(state.u_n), 99.5) + 1e-6
            im.set_clim(-vmax, vmax)
            im.set_data(state.u_n)

            if state.recording:
                state.record_frames.append(state.u_n.copy())

            state.t += dt
        return (im,)

    ani = FuncAnimation(fig, update, interval=15, blit=False)
    return fig, ani


def launch_analysis_window(state: SimState):
    fig, (ax_probe, ax_fft, ax_spec) = plt.subplots(3, 1, figsize=(8, 9))
    fig.suptitle("Analiza sygnału w sondzie")

    # Wykres czasowy
    line_probe, = ax_probe.plot([], [], lw=1.2)
    ax_probe.set_title("Sygnał w punkcie pomiarowym")
    ax_probe.set_xlabel("Czas [s]")
    ax_probe.set_ylabel("u")

    # Widmo FFT
    line_fft, = ax_fft.plot([], [], lw=1.2)
    ax_fft.set_title("Widmo amplitudy (FFT)")
    ax_fft.set_xlabel("Częstotliwość [Hz]")
    ax_fft.set_ylabel("Amplituda")

    # Spektrogram
    ax_spec.set_title("Spektrogram")
    ax_spec.set_xlabel("Czas [s]")
    ax_spec.set_ylabel("Częstotliwość [Hz]")

    def update(_frame):
        with state.lock:
            times = np.array(state.probe_times, dtype=float)
            if times.size >= 8:
                idxs = np.arange(state.probe_ptr - times.size, state.probe_ptr)
                vals = state.probe_buffer.take(idxs, mode='wrap')

                # Wykres czasowy
                line_probe.set_data(times, vals)
                ax_probe.set_xlim(times[0], times[-1])
                vspan = float(np.max(np.abs(vals)))
                vspan = max(1e-6, vspan)
                ax_probe.set_ylim(-1.2*vspan, 1.2*vspan)

                # FFT
                dt = state.compute_dt()
                win = np.hanning(len(vals))
                sigw = vals * win
                freqs = np.fft.rfftfreq(len(sigw), d=dt)
                amps = (2.0 / np.sum(win)) * np.abs(np.fft.rfft(sigw))
                line_fft.set_data(freqs, amps)
                ax_fft.set_xlim(0, max(2.5, max(state.f1, state.f2) * 3.0))
                ymax = float(np.percentile(amps, 99.5)) * 1.2 if np.any(amps > 0) else 1.0
                ax_fft.set_ylim(0, ymax)

                # Spektrogram
                if state.probe_ptr % 20 == 0:
                    ax_spec.cla()
                    ax_spec.set_title("Spektrogram")
                    ax_spec.set_xlabel("Czas [s]")
                    ax_spec.set_ylabel("Częstotliwość [Hz]")
                    Fs = 1.0 / dt
                    ax_spec.specgram(vals, NFFT=256, Fs=Fs, noverlap=192, cmap='magma')

        return (line_probe, line_fft)

    ani = FuncAnimation(fig, update, interval=300, blit=False)
    return fig, ani


def launch_control_window(state: SimState):
    root = tk.Tk()
    root.title("Sterowanie symulacją fal (Tkinter)")

    def add_slider(label, from_, to, getter, setter, fmt="{:.3f}"):
        frame = ttk.Frame(root)
        frame.pack(fill='x', padx=8, pady=4)

        ttk.Label(frame, text=label, width=22).pack(side='left')

        val_var = tk.DoubleVar(value=getter())
        val_label = ttk.Label(frame, text=fmt.format(val_var.get()), width=10, anchor='e')
        val_label.pack(side='right')

        def on_slide(v):
            v = float(v)
            with state.lock:
                setter(v)
            val_label.config(text=fmt.format(v))

        scale = ttk.Scale(frame, from_=from_, to=to, orient='horizontal', command=on_slide)
        scale.set(val_var.get())
        scale.pack(side='left', fill='x', expand=True, padx=8)

        return scale, val_label

    def add_button(text, cmd):
        def wrapped():
            with state.lock:
                cmd()
        btn = ttk.Button(root, text=text, command=wrapped)
        btn.pack(fill='x', padx=8, pady=4)
        return btn

    def add_radio(label, options, getter, setter):
        frame = ttk.Frame(root)
        frame.pack(fill='x', padx=8, pady=4)
        ttk.Label(frame, text=label, width=22).pack(side='left')
        var = tk.StringVar(value=getter())
        def on_change():
            with state.lock:
                setter(var.get())
        for opt in options:
            rb = ttk.Radiobutton(frame, text=opt.capitalize(), value=opt, variable=var, command=on_change)
            rb.pack(side='left', padx=4)
        return frame

    # Slidery parametrów
    add_slider("Częstotliwość f1 [Hz]", 0.05, 2.0, lambda: state.f1, lambda v: setattr(state, 'f1', v))
    add_slider("Częstotliwość f2 [Hz]", 0.05, 2.0, lambda: state.f2, lambda v: setattr(state, 'f2', v))
    add_slider("Amplituda A1", 0.0, 2.0, lambda: state.A1, lambda v: setattr(state, 'A1', v))
    add_slider("Amplituda A2", 0.0, 2.0, lambda: state.A2, lambda v: setattr(state, 'A2', v))
    add_slider("Faza phi1 [rad]", -np.pi, np.pi, lambda: state.phi1, lambda v: setattr(state, 'phi1', v), fmt="{:.2f}")
    add_slider("Faza phi2 [rad]", -np.pi, np.pi, lambda: state.phi2, lambda v: setattr(state, 'phi2', v), fmt="{:.2f}")
    add_slider("Rozstaw d", 0.1, 0.9*state.Lx, lambda: state.d, lambda v: setattr(state, 'd', v))
    add_slider("Prędkość c", 0.2, 3.0, lambda: state.c, lambda v: setattr(state, 'c', v))
    add_slider("Tłumienie gamma", 0.0, 0.1, lambda: state.gamma, lambda v: setattr(state, 'gamma', v), fmt="{:.4f}")

    add_radio("Ściany góra/dół", ['neumann', 'dirichlet'], lambda: state.bc, lambda v: setattr(state, 'bc', v))

    # 🎨 Wybór palety kolorów
    def add_cmap_selector():
        frame = ttk.Frame(root)
        frame.pack(fill='x', padx=8, pady=4)
        ttk.Label(frame, text="Paleta kolorów:", width=22).pack(side='left')
        cmap_list = sorted(plt.colormaps())
        var = tk.StringVar(value=state.cmap)
        def on_change(*_):
            with state.lock:
                state.cmap = var.get()
        dropdown = ttk.Combobox(frame, textvariable=var, values=cmap_list, state='readonly')
        dropdown.pack(side='left', fill='x', expand=True, padx=8)
        dropdown.bind("<<ComboboxSelected>>", on_change)
    add_cmap_selector()

    # 🎞 Przełącznik formatu zapisu
    def add_format_selector():
        frame = ttk.Frame(root)
        frame.pack(fill='x', padx=8, pady=4)
        ttk.Label(frame, text="Format zapisu:", width=22).pack(side='left')
        var = tk.StringVar(value=state.record_format)
        def on_change():
            with state.lock:
                state.record_format = var.get()
        for opt in ['gif', 'mp4']:
            rb = ttk.Radiobutton(frame, text=opt.upper(), value=opt, variable=var, command=on_change)
            rb.pack(side='left', padx=4)
    add_format_selector()

    # 🎥 Przycisk nagrywania
    def toggle_recording():
        with state.lock:
            if not state.recording:
                state.recording = True
                state.record_frames.clear()
            else:
                state.recording = False
                threading.Thread(target=save_recording, args=(state,), daemon=True).start()

    def save_recording(state):
        with state.lock:
            state.saving_status = "Trwa zapis..."
            state.saving_progress = 0

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"symulacja_{now}.{state.record_format}"

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(state.record_frames[0], extent=[0, state.Lx, 0, state.Ly],
                       origin='lower', cmap=state.cmap, interpolation='bilinear')
        ax.set_title("Zapisana animacja")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def update(i):
            im.set_data(state.record_frames[i])
            return (im,)

        ani = FuncAnimation(fig, update, frames=len(state.record_frames), interval=15, blit=False)

        try:
            if state.record_format == 'gif':
                writer = 'pillow'
            else:
                writer = 'ffmpeg'

            def on_frame(i):
                with state.lock:
                    state.saving_progress = i + 1

            ani.save(filename, writer=writer, fps=60, progress_callback=on_frame)
            print(f"✅ Zapisano animację do pliku: {filename}")
        except Exception as e:
            print(f"❌ Błąd zapisu: {e}")
        finally:
            plt.close(fig)
            with state.lock:
                state.saving_status = None
                state.saving_progress = 0

    add_button("Start / Stop nagrywania", toggle_recording)
    add_button("Start / Stop symulacji", lambda: setattr(state, 'running', not state.running))

    def do_reset():
        state.u_nm1.fill(0.0)
        state.u_n.fill(0.0)
        state.u_np1.fill(0.0)
        state.t = 0.0
        state.probe_buffer.fill(0.0)
        state.probe_ptr = 0
        state.probe_times.clear()
    add_button("Reset", do_reset)

    # 🔄 Pasek postępu zapisu
    progress_frame = ttk.Frame(root)
    progress_frame.pack(fill='x', padx=8, pady=4)
    status_label = ttk.Label(progress_frame, text="", width=22)
    status_label.pack(side='left')
    progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', length=200, mode='determinate')
    progress_bar.pack(side='left', fill='x', expand=True, padx=8)

    def update_progress():
        with state.lock:
            if state.saving_status:
                status_label.config(text=state.saving_status)
                progress_bar['maximum'] = len(state.record_frames)
                progress_bar['value'] = state.saving_progress
            else:
                status_label.config(text="")
                progress_bar['value'] = 0
        root.after(100, update_progress)
    update_progress()

    root.geometry("560x620")
    root.mainloop()


if __name__ == "__main__":
    state = SimState()

    # Panel sterowania w osobnym wątku
    threading.Thread(target=launch_control_window, args=(state,), daemon=True).start()

    # Okna Matplotlib w głównym wątku
    fig_anim, ani_anim = launch_animation_window(state)
    fig_an, ani_an = launch_analysis_window(state)

    plt.show()

