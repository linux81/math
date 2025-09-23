from simulation import SimulationState
from gui_pygame import run_pygame
from gui_tkinter import run_tkinter
import threading

if __name__ == "__main__":
    state = SimulationState()

    pygame_thread = threading.Thread(target=run_pygame, args=(state,), daemon=True)
    pygame_thread.start()

    run_tkinter(state)
    print("Uruchamiam Pygame i Tkinter...")

