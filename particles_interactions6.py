import pygame
import numpy as np
from collections import defaultdict

# --- PARAMETRY OKNA I SYMULACJI ---
WIDTH, HEIGHT = 900, 650
NUM_PARTICLES = 60
PARTICLE_RADIUS = 5
FPS = 60
dt = 1.0 / FPS

# --- GRAWITACJA GLOBALNA (opcjonalna) ---
GLOBAL_GRAVITY_Y = np.array([0.0, 0.05])
apply_global_gravity_y = False

# --- PARAMETRY FIZYCZNE ---
GRAVITATIONAL_CONSTANT = 0.5
SOFTENING_EPS2 = (PARTICLE_RADIUS * 0.75) ** 2
RESTITUTION = 0.9
WALL_RESTITUTION = 0.9

# --- SIATKA PRZESTRZENNA ---
CELL_SIZE = 2 * PARTICLE_RADIUS
GRID_COLS = max(1, WIDTH // CELL_SIZE)
GRID_ROWS = max(1, HEIGHT // CELL_SIZE)
spatial_grid = defaultdict(list)

# --- SŁOŃCA ---
SUN_ENABLED = True
SUN_POSITION = np.array([WIDTH/2, HEIGHT/2], dtype=float)
SUN_MASS_GLOBAL = 1500.0
sun_radius = 12

extra_suns = []  # lista dodatkowych słońc dodanych myszką
SUN_MASS_LOCAL = 5000.0

# --- INICJALIZACJA PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Symulacja cząstek — grawitacja, energia, histogramy")
font = pygame.font.SysFont("Arial", 16)
clock = pygame.time.Clock()

# --- STAN SYMULACJI ---
positions = None
velocities = None
masses = None
accelerations = None
selected_index = None

energy_history = []
show_histogram = False
show_velocity_histogram = False
show_energy_plot = False
step_count = 0
sim_time = 0.0

RANDOM_INITIAL_VELOCITY = True


def initialize_particle_state():
    pos = np.random.rand(NUM_PARTICLES, 2) * np.array([WIDTH, HEIGHT])
    if RANDOM_INITIAL_VELOCITY:
        angles = np.random.rand(NUM_PARTICLES) * 2 * np.pi
        speeds = np.random.uniform(20, 100.0, size=NUM_PARTICLES)
        vel = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, np.newaxis]
    else:
        vel = np.random.randn(NUM_PARTICLES, 2)
    mass = np.random.uniform(0.5, 2.0, size=NUM_PARTICLES)
    return pos.astype(float), vel.astype(float), mass.astype(float)


def reset_simulation():
    global positions, velocities, masses, accelerations, selected_index, energy_history, step_count, sim_time
    positions, velocities, masses = initialize_particle_state()
    accelerations = np.zeros((len(positions), 2), dtype=float)
    selected_index = None
    energy_history.clear()
    step_count = 0
    sim_time = 0.0


def mass_to_color(masses_arr):
    if masses_arr.size == 0:
        return []
    norm = (masses_arr - masses_arr.min()) / (masses_arr.max() - masses_arr.min() + 1e-6)
    R = (255 * norm).astype(np.uint8)
    G = (255 * (1 - norm)).astype(np.uint8)
    B = (128 + 127 * (1 - norm)).astype(np.uint8)
    return list(zip(R, G, B))


def build_spatial_grid():
    spatial_grid.clear()
    if positions is None:
        return
    N = len(positions)
    for i in range(N):
        x, y = positions[i]
        col = int(x // CELL_SIZE)
        row = int(y // CELL_SIZE)
        if 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS:
            spatial_grid[(col, row)].append(i)


def compute_accelerations_numpy():
    if positions is None or len(positions) == 0:
        return np.zeros((0, 2), dtype=float)
    N = len(positions)

    # Siły wzajemne między cząstkami
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2) + SOFTENING_EPS2
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    unit_vectors = diff / dist[:, :, np.newaxis]
    m_matrix = masses[:, np.newaxis] * masses[np.newaxis, :]
    force_mag = GRAVITATIONAL_CONSTANT * m_matrix / dist_sq
    forces = np.sum(force_mag[:, :, np.newaxis] * unit_vectors, axis=1)

    # Globalne słońce
    if SUN_ENABLED:
        diff_s = SUN_POSITION - positions
        dist_sq_s = np.sum(diff_s**2, axis=1) + SOFTENING_EPS2
        dist_s = np.sqrt(dist_sq_s)
        unit_s = diff_s / dist_s[:, np.newaxis]
        force_mag_s = GRAVITATIONAL_CONSTANT * masses * SUN_MASS_GLOBAL / dist_sq_s
        forces += force_mag_s[:, np.newaxis] * unit_s

    # Dodatkowe słońca
    for spos in extra_suns:
        diff_s2 = spos - positions
        dist_sq_s2 = np.sum(diff_s2**2, axis=1) + SOFTENING_EPS2
        dist_s2 = np.sqrt(dist_sq_s2)
        unit_s2 = diff_s2 / dist_s2[:, np.newaxis]
        force_mag_s2 = GRAVITATIONAL_CONSTANT * masses * SUN_MASS_LOCAL / dist_sq_s2
        forces += force_mag_s2[:, np.newaxis] * unit_s2

    acc = forces / masses[:, np.newaxis]
    if apply_global_gravity_y:
        acc += GLOBAL_GRAVITY_Y
    return acc.astype(float)


def handle_walls():
    if positions is None or len(positions) == 0:
        return
    N = len(positions)
    for i in range(N):
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        # Lewa/prawa
        if positions[i, 0] < radius:
            positions[i, 0] = radius
            velocities[i, 0] = -WALL_RESTITUTION * velocities[i, 0]
        elif positions[i, 0] > WIDTH - radius:
            positions[i, 0] = WIDTH - radius
            velocities[i, 0] = -WALL_RESTITUTION * velocities[i, 0]
        # Góra/dół
        if positions[i, 1] < radius:
            positions[i, 1] = radius
            velocities[i, 1] = -WALL_RESTITUTION * velocities[i, 1]
        elif positions[i, 1] > HEIGHT - radius:
            positions[i, 1] = HEIGHT - radius
            velocities[i, 1] = -WALL_RESTITUTION * velocities[i, 1]


def handle_sun_collision():
    if positions is None or len(positions) == 0:
        return
    N = len(positions)

    # Globalne słońce
    if SUN_ENABLED:
        for i in range(N):
            dx = positions[i] - SUN_POSITION
            dist = np.linalg.norm(dx)
            radius_i = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
            min_dist = sun_radius + radius_i
            if dist < min_dist and dist > 1e-6:
                n = dx / dist
                overlap = min_dist - dist
                positions[i] += n * overlap
                velocities[i] -= 2 * np.dot(velocities[i], n) * n
                velocities[i] *= RESTITUTION

    # Dodatkowe słońca
    for spos in extra_suns:
        for i in range(N):
            dx = positions[i] - spos
            dist = np.linalg.norm(dx)
            radius_i = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
            min_dist = sun_radius + radius_i
            if dist < min_dist and dist > 1e-6:
                n = dx / dist
                overlap = min_dist - dist
                positions[i] += n * overlap
                velocities[i] -= 2 * np.dot(velocities[i], n) * n
                velocities[i] *= RESTITUTION

                
                
def add_particle_at(pos):
    global positions, velocities, masses, accelerations
    new_pos = np.array(pos, dtype=float)
    angle = np.random.rand() * 2 * np.pi
    speed = np.random.uniform(1.0, 3.0)
    velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
    mass = np.random.uniform(0.5, 2.0)

    if positions is None or len(positions) == 0:
        positions = np.array([new_pos], dtype=float)
        velocities = np.array([velocity], dtype=float)
        masses = np.array([mass], dtype=float)
        accelerations = np.zeros((1, 2), dtype=float)
    else:
        positions = np.vstack([positions, new_pos])
        velocities = np.vstack([velocities, velocity])
        masses = np.append(masses, mass)
        accelerations = np.vstack([accelerations, np.zeros(2)])

    build_spatial_grid()
    accelerations = compute_accelerations_numpy()


def remove_particle_at(pos):
    global positions, velocities, masses, accelerations, selected_index
    if positions is None or len(positions) == 0:
        return
    click = np.array(pos, dtype=float)
    N = len(positions)
    for i in range(N):
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        if np.linalg.norm(positions[i] - click) < radius + 3:
            positions = np.delete(positions, i, axis=0)
            velocities = np.delete(velocities, i, axis=0)
            masses = np.delete(masses, i)
            accelerations = np.delete(accelerations, i, axis=0)
            if selected_index is not None:
                if selected_index == i:
                    selected_index = None
                elif selected_index > i:
                    selected_index -= 1
            break
    build_spatial_grid()
    if positions is not None and len(positions) > 0:
        accelerations = compute_accelerations_numpy()
    else:
        accelerations = np.zeros((0, 2), dtype=float)


def add_sun_at(pos):
    global extra_suns
    extra_suns.append(np.array(pos, dtype=float))


def remove_sun_at(pos):
    global extra_suns
    if not extra_suns:
        return
    click = np.array(pos, dtype=float)
    for i, spos in enumerate(extra_suns):
        if np.linalg.norm(spos - click) < sun_radius + 3:
            extra_suns.pop(i)
            break


def compute_energy():
    if positions is None or len(positions) == 0:
        return 0.0
    # energia kinetyczna
    kinetic = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

    # energia potencjalna cząstek między sobą
    potential = 0.0
    N = len(positions)
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-6:
                potential -= GRAVITATIONAL_CONSTANT * masses[i] * masses[j] / r

    # energia potencjalna od słońc
    if SUN_ENABLED:
        for i in range(N):
            r = np.linalg.norm(positions[i] - SUN_POSITION)
            if r > 1e-6:
                potential -= GRAVITATIONAL_CONSTANT * masses[i] * SUN_MASS_GLOBAL / r

    for spos in extra_suns:
        for i in range(N):
            r = np.linalg.norm(positions[i] - spos)
            if r > 1e-6:
                potential -= GRAVITATIONAL_CONSTANT * masses[i] * SUN_MASS_LOCAL / r

    return float(kinetic + potential)


def draw_histogram():
    # histogram mas cząstek w prawym dolnym rogu
    if positions is None or len(masses) == 0:
        return
    counts, bins = np.histogram(masses, bins=12, range=(0.5, 2.0))
    max_count = counts.max() if counts.size > 0 else 0
    base_x = WIDTH - 140
    base_y = HEIGHT - 20

    # ramka i tytuł
    pygame.draw.rect(screen, (80, 80, 80), (base_x - 10, base_y - 100, 130, 100), 1)
    title = font.render("Histogram mas (H)", True, (180, 220, 255))
    screen.blit(title, (base_x - 8, base_y - 120))

    for i, c in enumerate(counts):
        x = base_x + i * 10
        h = int(80 * c / max_count) if max_count > 0 else 0
        pygame.draw.rect(screen, (100, 200, 255), (x, base_y - h, 8, h))


def draw_velocity_histogram():
    # histogram prędkości cząstek w prawym dolnym rogu (bardziej na lewo od mas)
    if positions is None or len(velocities) == 0:
        return
    speeds = np.linalg.norm(velocities, axis=1)
    vmax = max(1e-6, speeds.max())
    counts, bins = np.histogram(speeds, bins=12, range=(0.0, vmax))
    max_count = counts.max() if counts.size > 0 else 0
    base_x = WIDTH - 300
    base_y = HEIGHT - 20

    # ramka i tytuł
    pygame.draw.rect(screen, (80, 60, 100), (base_x - 10, base_y - 100, 130, 100), 1)
    title = font.render("Histogram prędkości (V)", True, (200, 150, 255))
    screen.blit(title, (base_x - 8, base_y - 120))

    for i, c in enumerate(counts):
        x = base_x + i * 10
        h = int(80 * c / max_count) if max_count > 0 else 0
        pygame.draw.rect(screen, (200, 150, 255), (x, base_y - h, 8, h))


def draw_energy_plot():
    # wykres energii w prawym górnym rogu
    if len(energy_history) < 2:
        return
    max_e = max(energy_history)
    min_e = min(energy_history)
    scale = 100 / (max_e - min_e + 1e-6)
    base_x = WIDTH - 250
    base_y = 140

    # ramka i tytuł
    pygame.draw.rect(screen, (80, 80, 80), (base_x - 10, 30, 220, 110), 1)
    title = font.render("Energia (E)", True, (255, 255, 0))
    screen.blit(title, (base_x - 8, 8))

    # linia energii
    # rysujemy ostatnie 200 punktów (lub mniej)
    span = min(200, len(energy_history))
    start = len(energy_history) - span
    for i in range(start + 1, len(energy_history)):
        x1 = base_x + (i - 1 - start)
        y1 = base_y - int((energy_history[i - 1] - min_e) * scale)
        x2 = base_x + (i - start)
        y2 = base_y - int((energy_history[i] - min_e) * scale)
        pygame.draw.line(screen, (255, 255, 0), (x1, y1), (x2, y2), 1)


def draw_particles():
    if positions is None or len(positions) == 0:
        return
    colors = mass_to_color(masses)
    N = len(positions)

    # cząstki
    for i in range(N):
        x, y = positions[i]
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        pygame.draw.circle(screen, colors[i], (int(x), int(y)), radius)

    # info dla wybranej cząstki (na wierzchu)
    if selected_index is not None and 0 <= selected_index < N:
        x, y = positions[selected_index]
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[selected_index]))
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), radius + 2, 1)

        vx, vy = velocities[selected_index]
        speed = float(np.linalg.norm([vx, vy]))
        mass_val = float(masses[selected_index])

        # wektor prędkości
        pygame.draw.line(screen, (255, 255, 0), (x, y), (x + vx * 10, y + vy * 10), 2)

        # etykieta
        label = font.render(f"v={speed:.2f}, m={mass_val:.2f}", True, (255, 255, 255))
        screen.blit(label, (int(x) + 10, int(y) - 20))

    # słońca
    if SUN_ENABLED:
        pygame.draw.circle(screen, (255, 200, 0), SUN_POSITION.astype(int), sun_radius)
    for spos in extra_suns:
        pygame.draw.circle(screen, (255, 150, 0), spos.astype(int), sun_radius)

        
        
       # --- START SYMULACJI ---
reset_simulation()
build_spatial_grid()
accelerations = compute_accelerations_numpy()

running = True
while running:
    screen.fill((25, 25, 30))

    # --- Obsługa zdarzeń ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                reset_simulation()
                build_spatial_grid()
                accelerations = compute_accelerations_numpy()
            elif event.key == pygame.K_g:
                apply_global_gravity_y = not apply_global_gravity_y
            elif event.key == pygame.K_h:
                show_histogram = not show_histogram
            elif event.key == pygame.K_v:
                show_velocity_histogram = not show_velocity_histogram
            elif event.key == pygame.K_e:
                show_energy_plot = not show_energy_plot
            elif event.key == pygame.K_s:
                SUN_ENABLED = not SUN_ENABLED  # globalne słońce ON/OFF

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # lewy przycisk
                clicked_index = None
                if positions is not None and len(positions) > 0:
                    for i in range(len(positions)):
                        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
                        if np.linalg.norm(positions[i] - np.array(event.pos, dtype=float)) < radius + 3:
                            clicked_index = i
                            break
                if clicked_index is not None:
                    selected_index = clicked_index  # zaznacz cząstkę
                else:
                    # Shift + lewy klik → dodaj słońce, inaczej → cząstkę
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        add_sun_at(event.pos)
                    else:
                        add_particle_at(event.pos)

            elif event.button == 3:  # prawy przycisk → usuń
                remove_particle_at(np.array(event.pos, dtype=float))
                remove_sun_at(np.array(event.pos, dtype=float))

    # --- Integracja (Velocity Verlet) ---
    if positions is not None and len(positions) > 0:
        positions += velocities * dt + 0.5 * accelerations * dt**2
        build_spatial_grid()
        new_acc = compute_accelerations_numpy()
        velocities += 0.5 * (accelerations + new_acc) * dt
        accelerations = new_acc

    step_count += 1
    sim_time += dt

    # --- Kolizje i odbicia ---
    if positions is not None and len(positions) > 0:
        handle_walls()
        handle_sun_collision()

    # --- Energia układu (obliczana co klatkę) ---
    energy = compute_energy()
    energy_history.append(energy)
    if len(energy_history) > 600:  # bufor historii
        energy_history.pop(0)

    # --- Rysowanie cząstek i słońc ---
    draw_particles()

    # --- UI i etykiety ---
    grav_status = "ON" if apply_global_gravity_y else "OFF"
    sun_status = "ON" if SUN_ENABLED else "OFF"

    grav_label = font.render(f"Global Gravity (G): {grav_status}", True, (255, 255, 255))
    sun_label = font.render(f"Global Sun (S): {sun_status}", True, (255, 255, 255))
    reset_label = font.render("Reset (R)", True, (255, 255, 255))
    step_label = font.render(f"Krok: {step_count}", True, (255, 255, 255))
    time_label = font.render(f"Czas: {sim_time:.2f} s", True, (255, 255, 255))
    energy_label = font.render(f"Energia układu: {energy:.2f}", True, (255, 255, 255))

    screen.blit(grav_label, (10, 10))
    screen.blit(sun_label, (10, 30))
    screen.blit(reset_label, (10, 50))
    screen.blit(step_label, (10, 70))
    screen.blit(time_label, (10, 90))
    screen.blit(energy_label, (10, 110))

    legend = font.render(
        "Lewy=cząstka | Shift+Lewy=słońce | Prawy=usuń | H=histogram mas | V=histogram prędkości | E=energia | G=grawitacja",
        True, (200, 200, 200)
    )
    screen.blit(legend, (10, HEIGHT - 30))

    # --- Dodatkowe wizualizacje (dynamiczne) ---
    if show_histogram:
        draw_histogram()
    if show_velocity_histogram:
        draw_velocity_histogram()
    if show_energy_plot:
        draw_energy_plot()

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
 

