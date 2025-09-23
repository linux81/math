import pygame
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import pygame.surfarray

def draw_grid(screen, state):
    """Renderuje siatkę temperatury jako obraz RGB z użyciem cmap."""
    cmap = cm.get_cmap(state.current_cmap)
    norm = colors.Normalize(vmin=-150, vmax=150)
    rgb_array = (255 * cmap(norm(state.u))[:, :, :3]).astype(np.uint8)
    surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
    scaled_surface = pygame.transform.scale(surface, screen.get_size())
    screen.blit(scaled_surface, (0, 0))

def run_pygame(state):
    pygame.init()
    cell_size = 10
    last_size = (state.nx, state.ny)
    screen = pygame.display.set_mode((state.nx * cell_size, state.ny * cell_size))
    clock = pygame.time.Clock()
    print("Pygame loop running...")

    while True:
        # Sprawdź, czy rozmiar siatki się zmienił
        if (state.nx, state.ny) != last_size:
            screen = pygame.display.set_mode((state.nx * cell_size, state.ny * cell_size))
            last_size = (state.nx, state.ny)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                screen_width, screen_height = screen.get_size()
                cell_width = screen_width / state.ny
                cell_height = screen_height / state.nx

                j = int(event.pos[0] / cell_width)
                i = int(event.pos[1] / cell_height)

                if 0 <= i < state.nx and 0 <= j < state.ny:
                    if event.button == 1:
                        state.add_source(i, j, state.heat_temp)
                    elif event.button == 3:
                        state.add_source(i, j, -state.cooling_strength)



        state.update()
        draw_grid(screen, state)
        draw_legend(screen, state)
        draw_boundary_overlay(screen, state)
        mouse_pos = pygame.mouse.get_pos()
        draw_temperature_under_cursor(screen, state, mouse_pos)

        pygame.display.flip()
        clock.tick(int(30 * state.speed_factor))
        fps = clock.get_fps()
        sim_time = state.frame_counter * 0.1
        pygame.display.set_caption(f"Symulacja — FPS: {fps:.1f} | Czas: {sim_time:.1f}s")

        
def draw_boundary_overlay(screen, state):
    color_map = {
        "Neumann": (0, 0, 255),
        "Dirichlet": (255, 0, 0),
        "Periodic": (0, 255, 0)
    }
    w, h = screen.get_size()
    thickness = 4

    pygame.draw.rect(screen, color_map[state.boundary_types["top"]], (0, 0, w, thickness))
    pygame.draw.rect(screen, color_map[state.boundary_types["bottom"]], (0, h-thickness, w, thickness))
    pygame.draw.rect(screen, color_map[state.boundary_types["left"]], (0, 0, thickness, h))
    pygame.draw.rect(screen, color_map[state.boundary_types["right"]], (w-thickness, 0, thickness, h))

def draw_legend(screen, state):
    cmap = cm.get_cmap(state.current_cmap)
    norm = colors.Normalize(vmin=-150, vmax=150)

    legend_width = 20
    legend_height = screen.get_height()

    # Odwrócony gradient: wysoka temperatura na górze
    gradient = np.linspace(150, -150, legend_height)
    color_array = (255 * cmap(norm(gradient))[:, :3]).astype(np.uint8)

    # Tworzymy obraz RGB
    rgb_image = np.repeat(color_array[:, np.newaxis, :], legend_width, axis=1)
    surface = pygame.surfarray.make_surface(np.transpose(rgb_image, (1, 0, 2)))
    screen.blit(surface, (screen.get_width() - legend_width - 10, 0))

    # Etykiety temperatury — dopasowane do odwróconego gradientu
    font = pygame.font.SysFont(None, 20)
    for temp, y in [(150, 0), (0, legend_height // 2), (-150, legend_height - 20)]:
        label = font.render(f"{temp}°C", True, (255, 255, 255))
        screen.blit(label, (screen.get_width() - legend_width - 50, y))

def draw_temperature_under_cursor(screen, state, mouse_pos):
    screen_width, screen_height = screen.get_size()
    cell_width = screen_width / state.ny      # NY → poziom (kolumny)
    cell_height = screen_height / state.nx    # NX → pion (wiersze)

    j = int(mouse_pos[0] / cell_width)  # kolumna
    i = int(mouse_pos[1] / cell_height) # wiersz

    if 0 <= i < state.nx and 0 <= j < state.ny:
        temp = state.u[i, j]  # NumPy: [wiersz, kolumna]
        font = pygame.font.SysFont(None, 24)
        label = font.render(f"({i}, {j}) — {temp:.1f}°C", True, (255, 255, 255))
        screen.blit(label, (mouse_pos[0] + 10, mouse_pos[1] + 10))





