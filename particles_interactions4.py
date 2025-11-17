import pygame
import numpy as np
from collections import defaultdict

# --- PARAMETRY ---
WIDTH, HEIGHT = 800, 600
NUM_PARTICLES = 50
PARTICLE_RADIUS = 5
HIST_WIDTH = 200
HIST_HEIGHT = int(HEIGHT * 0.4)
FPS = 30
dt = 1.0 / FPS

# Globalna Grawitacja (kierunek Y)
GLOBAL_GRAVITY_Y = np.array([0.0, 0.05])
apply_global_gravity_y = False

# Grawitacja między cząstkami
GRAVITATIONAL_CONSTANT = 0.5
SOFTENING_EPS2 = (PARTICLE_RADIUS * 0.75) ** 2
MIN_DISTANCE_SQ = 100.0

# Kolizje
RESTITUTION = 0.9
PENETRATION_SLOP = 0.01
PENETRATION_PERCENT = 0.2

# Odbicia od ścian
WALL_RESTITUTION = 0.9

# --- PARAMETRY SIATKI ---
CELL_SIZE = 2 * PARTICLE_RADIUS
GRID_COLS = max(1, WIDTH // CELL_SIZE)
GRID_ROWS = max(1, HEIGHT // CELL_SIZE)
spatial_grid = defaultdict(list)

# --- INICJALIZACJA ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("Arial", 16)
clock = pygame.time.Clock()

# Stan początkowy
initial_positions = None
initial_velocities = None
initial_masses = None

positions = None
velocities = None
masses = None
accelerations = None
selected_index = None

energy_history = []
show_histogram = False
show_energy_plot = False
step_count = 0
sim_time = 0.0

RANDOM_INITIAL_VELOCITY = True   # ustaw na False, aby wyłączyć losowe prędkości

SUN_ENABLED = True
SUN_POSITION = np.array([WIDTH/2, HEIGHT/2])   # środek ekranu
SUN_MASS = 1500.0                               # duża masa




# --- FUNKCJE STANU ---

def initialize_particle_state():
    pos = np.random.rand(NUM_PARTICLES, 2) * np.array([WIDTH, HEIGHT])
    
    if RANDOM_INITIAL_VELOCITY:
        # losowy kierunek (kąt) i losowa prędkość
        angles = np.random.rand(NUM_PARTICLES) * 2 * np.pi
        speeds = np.random.uniform(5, 50.0, size=NUM_PARTICLES)  # zakres prędkości
        vel = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, np.newaxis]
    else:
        # standardowe losowe prędkości (np. rozkład normalny)
        vel = np.random.randn(NUM_PARTICLES, 2)
    
    mass = np.random.uniform(0.5, 2.0, size=NUM_PARTICLES)
    return pos, vel, mass


def reset_simulation():
    global positions, velocities, masses, accelerations
    global initial_positions, initial_velocities, initial_masses, energy_history, selected_index

    # zawsze generuj nowe cząstki
    initial_positions, initial_velocities, initial_masses = initialize_particle_state()

    positions = initial_positions.copy()
    velocities = initial_velocities.copy()
    masses = initial_masses.copy()
    accelerations = np.zeros((NUM_PARTICLES, 2))
    selected_index = None
    energy_history.clear()


def velocity_magnitude(v):
    return np.linalg.norm(v, axis=1)

def mass_to_color(masses):
    # kolor zależny od masy: lekkie bardziej zielono-niebieskie, ciężkie bardziej czerwone
    norm = (masses - masses.min()) / (masses.max() - masses.min() + 1e-6)
    R = (255 * norm).astype(np.uint8)
    G = (255 * (1 - norm)).astype(np.uint8)
    B = (128 + 127 * (1 - norm)).astype(np.uint8)
    return list(zip(R, G, B))

def compute_total_energy():
    # energia kinetyczna
    kinetic = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    # energia potencjalna (softened)
    potential = 0.0
    for i in range(NUM_PARTICLES):
        mi = masses[i]
        ri = positions[i]
        for j in range(i + 1, NUM_PARTICLES):
            mj = masses[j]
            rj = positions[j]
            r = rj - ri
            dist = np.linalg.norm(r)
            if dist > 1e-6:
                potential -= GRAVITATIONAL_CONSTANT * mi * mj / np.sqrt(dist**2 + SOFTENING_EPS2)
    # potencjał pola globalnego Y
    global_potential = 0.0
    if apply_global_gravity_y:
        g = GLOBAL_GRAVITY_Y[1]
        global_potential = np.sum(masses * g * positions[:, 1])
    return kinetic + potential + global_potential


# --- RYSOWANIE I UI ---

def draw_particles():
    colors = mass_to_color(masses)
    for i in range(NUM_PARTICLES):
        x, y = positions[i]
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        pygame.draw.circle(screen, colors[i], (int(x), int(y)), radius)
        if i == selected_index:
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), radius + 2, 1)
            vx, vy = velocities[i]
            speed = np.linalg.norm([vx, vy])
            mass = masses[i]
            pygame.draw.line(screen, (255, 255, 0), (x, y), (x + vx * 10, y + vy * 10), 2)
            label = font.render(f"v = {speed:.2f}, m = {mass:.2f}", True, (255, 255, 255))
            screen.blit(label, (x + 10, y - 20))

def draw_histogram():
    speeds = velocity_magnitude(velocities)
    hist, bins = np.histogram(speeds, bins=20, range=(0, max(10.0, speeds.max())))
    max_count = np.max(hist) if np.max(hist) > 0 else 1
    bin_width = HIST_WIDTH // len(hist)
    x_offset = WIDTH - HIST_WIDTH
    y_offset = HEIGHT - HIST_HEIGHT
    pygame.draw.rect(screen, (20, 20, 20), (x_offset, y_offset, HIST_WIDTH, HIST_HEIGHT))
    for i, count in enumerate(hist):
        h = int((count / max_count) * HIST_HEIGHT)
        x = x_offset + i * bin_width
        pygame.draw.rect(screen, (100, 200, 255), (x, y_offset + HIST_HEIGHT - h, bin_width - 2, h))

def draw_energy_plot():
    if len(energy_history) < 2:
        return
    x_offset = 10
    y_offset = HEIGHT - HIST_HEIGHT - 10
    pygame.draw.rect(screen, (20, 20, 20), (x_offset, y_offset, HIST_WIDTH, HIST_HEIGHT))
    min_e, max_e = min(energy_history), max(energy_history)
    if max_e == min_e: max_e += 1.0
    scale_x = HIST_WIDTH / (len(energy_history) - 1)
    scale_y = HIST_HEIGHT / (max_e - min_e)
    for i in range(1, len(energy_history)):
        x1 = x_offset + (i - 1) * scale_x
        y1 = y_offset + HIST_HEIGHT - (energy_history[i - 1] - min_e) * scale_y
        x2 = x_offset + i * scale_x
        y2 = y_offset + HIST_HEIGHT - (energy_history[i] - min_e) * scale_y
        pygame.draw.line(screen, (255, 100, 100), (x1, y1), (x2, y2), 2)
    label = font.render("Energy", True, (255, 255, 255))
    screen.blit(label, (x_offset, y_offset - 20))

def draw_mass_legend():
    legend_width = 150
    legend_height = 20
    x_offset = WIDTH - legend_width - 10
    y_offset = 10
    for i in range(legend_width):
        norm = i / legend_width
        color = (int(255 * norm), int(255 * (1 - norm)), int(128 + 127 * (1 - norm)))
        pygame.draw.line(screen, color, (x_offset + i, y_offset), (x_offset + i, y_offset + legend_height))
    label = font.render("Masa: lekka → ciężka", True, (255, 255, 255))
    screen.blit(label, (x_offset, y_offset + legend_height + 5))

def handle_click(pos):
    global selected_index
    mx, my = pos
    for i in range(NUM_PARTICLES):
        x, y = positions[i]
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        if np.hypot(mx - x, my - y) < radius + 3:
            selected_index = i
            return
    selected_index = None

# --- SIATKA ---

def build_spatial_grid():
    spatial_grid.clear()
    for i in range(NUM_PARTICLES):
        x, y = positions[i]
        col = int(x // CELL_SIZE)
        row = int(y // CELL_SIZE)
        if 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS:
            spatial_grid[(col, row)].append(i)

            
            
# --- AKCELERACJE (siły grawitacji) ---

def compute_accelerations():
    forces = np.zeros((NUM_PARTICLES, 2))
    for col in range(GRID_COLS):
        for row in range(GRID_ROWS):
            candidates = []
            for dc in [-1, 0, 1]:
                for dr in [-1, 0, 1]:
                    nc, nr = col + dc, row + dr
                    if 0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS:
                        if (nc, nr) in spatial_grid:
                            candidates.extend(spatial_grid[(nc, nr)])
            n_local = len(candidates)
            for a in range(n_local):
                i = candidates[a]
                mi = masses[i]
                ri = positions[i]
                for b in range(a + 1, n_local):
                    j = candidates[b]
                    mj = masses[j]
                    rj = positions[j]

                    r = rj - ri
                    dist_sq = np.dot(r, r) + SOFTENING_EPS2
                    dist_sq = max(dist_sq, MIN_DISTANCE_SQ)
                    r_norm = np.sqrt(dist_sq)
                    if r_norm == 0.0:
                        continue
                    r_unit = r / r_norm

                    force_mag = (GRAVITATIONAL_CONSTANT * mi * mj) / dist_sq
                    force = force_mag * r_unit
                    forces[i] += force
                    forces[j] -= force

    if SUN_ENABLED:
            diff = SUN_POSITION - positions
            dist_sq = np.sum(diff**2, axis=1) + SOFTENING_EPS2
            dist = np.sqrt(dist_sq)
            unit_vectors = diff / dist[:, np.newaxis]
            force_mag = GRAVITATIONAL_CONSTANT * masses * SUN_MASS / dist_sq
            forces += force_mag[:, np.newaxis] * unit_vectors

    acc = forces / masses[:, np.newaxis]
    if apply_global_gravity_y:
        acc += GLOBAL_GRAVITY_Y
    return acc


def compute_accelerations_numpy():
    # różnice pozycji między wszystkimi parami cząstek
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape (N,N,2)
    dist_sq = np.sum(diff**2, axis=2) + SOFTENING_EPS2               # shape (N,N)

    # maska, żeby nie liczyć i=j
    np.fill_diagonal(dist_sq, np.inf)

    # jednostkowe wektory kierunku
    dist = np.sqrt(dist_sq)
    unit_vectors = diff / dist[:, :, np.newaxis]

    # siły grawitacyjne: F = G * m_i * m_j / r^2
    m_matrix = masses[:, np.newaxis] * masses[np.newaxis, :]
    force_mag = GRAVITATIONAL_CONSTANT * m_matrix / dist_sq

    # wektory sił
    forces = np.sum(force_mag[:, :, np.newaxis] * unit_vectors, axis=1)

    # przyspieszenia
    acc = forces / masses[:, np.newaxis]

    # dodaj globalną grawitację Y (jeśli włączona)
    if apply_global_gravity_y:
        acc += GLOBAL_GRAVITY_Y

    return acc



# --- KOLIZJE ---

def handle_collisions():
    for col in range(GRID_COLS):
        for row in range(GRID_ROWS):
            candidates = []
            for dc in [-1, 0, 1]:
                for dr in [-1, 0, 1]:
                    nc = col + dc
                    nr = row + dr
                    if 0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS:
                        if (nc, nr) in spatial_grid:
                            candidates.extend(spatial_grid[(nc, nr)])

            n_local = len(candidates)
            for a in range(n_local):
                i = candidates[a]
                for b in range(a + 1, n_local):
                    j = candidates[b]

                    dx = positions[j] - positions[i]
                    dist = np.linalg.norm(dx)

                    radius_i = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
                    radius_j = int(PARTICLE_RADIUS * np.sqrt(masses[j]))
                    min_dist = radius_i + radius_j

                    if dist < min_dist and dist > 1e-6:
                        n = dx / dist
                        overlap = min_dist - dist

                        mi, mj = masses[i], masses[j]
                        inv_mi, inv_mj = 1.0 / mi, 1.0 / mj
                        total_inv_mass = inv_mi + inv_mj

                        # Korekcja penetracji (ze slopem i procentową korektą)
                        correction = max(overlap - PENETRATION_SLOP, 0.0) * PENETRATION_PERCENT
                        if total_inv_mass > 0.0 and correction > 0.0:
                            positions[i] -= (correction * inv_mi / total_inv_mass) * n
                            positions[j] += (correction * inv_mj / total_inv_mass) * n

                        # Impuls sprężysty
                        dv = velocities[i] - velocities[j]
                        rel_vel_n = np.dot(dv, n)
                        if rel_vel_n < 0.0:
                            j_impulse = -(1.0 + RESTITUTION) * rel_vel_n
                            j_impulse /= (inv_mi + inv_mj)
                            impulse_vec = j_impulse * n
                            velocities[i] += impulse_vec * inv_mi
                            velocities[j] -= impulse_vec * inv_mj

# --- ODBICIA OD ŚCIAN ---

def handle_walls():
    for i in range(NUM_PARTICLES):
        # X
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        if positions[i, 0] < radius:
            positions[i, 0] = radius
            velocities[i, 0] = -WALL_RESTITUTION * velocities[i, 0]
        elif positions[i, 0] > WIDTH - radius:
            positions[i, 0] = WIDTH - radius
            velocities[i, 0] = -WALL_RESTITUTION * velocities[i, 0]
        # Y
        if positions[i, 1] < radius:
            positions[i, 1] = radius
            velocities[i, 1] = -WALL_RESTITUTION * velocities[i, 1]
        elif positions[i, 1] > HEIGHT - radius:
            positions[i, 1] = HEIGHT - radius
            velocities[i, 1] = -WALL_RESTITUTION * velocities[i, 1]
            


            
def draw_sun():
    if SUN_ENABLED:
        pygame.draw.circle(screen, (255, 200, 0), SUN_POSITION.astype(int), 12)
        
        
def handle_sun_collision():
    if not SUN_ENABLED:
        return
    sun_radius = 12  # promień graficzny słońca
    for i in range(NUM_PARTICLES):
        dx = positions[i] - SUN_POSITION
        dist = np.linalg.norm(dx)
        radius_i = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        min_dist = sun_radius + radius_i
        if dist < min_dist and dist > 1e-6:
            n = dx / dist
            overlap = min_dist - dist
            positions[i] += n * overlap  # przesunięcie cząstki na zewnątrz
            # odbicie prędkości
            velocities[i] -= 2 * np.dot(velocities[i], n) * n
            velocities[i] *= RESTITUTION


            # --- GŁÓWNA PĘTLA ---
reset_simulation()
build_spatial_grid()
accelerations = compute_accelerations()

running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            handle_click(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                reset_simulation()
                build_spatial_grid()
                accelerations = compute_accelerations()
            elif event.key == pygame.K_g:
                apply_global_gravity_y = not apply_global_gravity_y
            elif event.key == pygame.K_h:
                show_histogram = not show_histogram
            elif event.key == pygame.K_e:
                show_energy_plot = not show_energy_plot
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:   # klawisz S włącza/wyłącza słońce
                    SUN_ENABLED = not SUN_ENABLED

    # --- Velocity Verlet ---
    positions += velocities * dt + 0.5 * accelerations * dt**2
    build_spatial_grid()
    new_acc = compute_accelerations_numpy()
    velocities += 0.5 * (accelerations + new_acc) * dt
    accelerations = new_acc
    step_count += 1
    sim_time += dt

    # Odbicia i kolizje
    handle_walls()
    handle_collisions()
    
    handle_sun_collision()
    # Rysowanie
    draw_particles()
    draw_mass_legend()
    
    draw_sun()

    # Histogram i energia
    if show_histogram:
        draw_histogram()

    total_energy = compute_total_energy()
    energy_history.append(total_energy)
    if len(energy_history) > WIDTH:
        energy_history.pop(0)

    energy_label = font.render(f"Energy: {total_energy:.2f}", True, (255, 255, 255))
    screen.blit(energy_label, (10, 50))
    if show_energy_plot:
        draw_energy_plot()

    # UI statusy
    grav_status = "ON" if apply_global_gravity_y else "OFF"
    grav_label = font.render(f"Global Gravity (G): {grav_status}", True, (255, 255, 255))
    screen.blit(grav_label, (10, 10))
    reset_label = font.render("Reset (R)", True, (255, 255, 255))
    screen.blit(reset_label, (10, 30))
    hist_status = "ON" if show_histogram else "OFF"
    energy_status = "ON" if show_energy_plot else "OFF"
    hist_label = font.render(f"Histogram (H): {hist_status}", True, (255, 255, 255))
    energy_label2 = font.render(f"Energy Plot (E): {energy_status}", True, (255, 255, 255))
    screen.blit(hist_label, (10, 70))
    screen.blit(energy_label2, (10, 90))
    step_label = font.render(f"Krok: {step_count}", True, (255, 255, 255))
    time_label = font.render(f"Czas: {sim_time:.2f} s", True, (255, 255, 255))
    screen.blit(step_label, (10, 110))
    screen.blit(time_label, (10, 130))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()

