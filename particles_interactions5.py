import pygame
import numpy as np
from collections import defaultdict

# --- PARAMETRY ---
WIDTH, HEIGHT = 800, 600
NUM_PARTICLES = 50
PARTICLE_RADIUS = 5
FPS = 30
dt = 1.0 / FPS

GLOBAL_GRAVITY_Y = np.array([0.0, 0.05])
apply_global_gravity_y = False

GRAVITATIONAL_CONSTANT = 0.5
SOFTENING_EPS2 = (PARTICLE_RADIUS * 0.75) ** 2
RESTITUTION = 0.9
WALL_RESTITUTION = 0.9

CELL_SIZE = 2 * PARTICLE_RADIUS
GRID_COLS = max(1, WIDTH // CELL_SIZE)
GRID_ROWS = max(1, HEIGHT // CELL_SIZE)
spatial_grid = defaultdict(list)

# --- SŁOŃCA ---
SUN_ENABLED = True
SUN_POSITION = np.array([WIDTH/2, HEIGHT/2])
SUN_MASS_GLOBAL = 1500.0
sun_radius = 12

extra_suns = []  # lista dodatkowych słońc dodanych myszką
SUN_MASS_LOCAL = 500.0

# --- INICJALIZACJA ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("Arial", 16)
clock = pygame.time.Clock()

positions = None
velocities = None
masses = None
accelerations = None
selected_index = None

energy_history = []
step_count = 0
sim_time = 0.0

RANDOM_INITIAL_VELOCITY = True



def initialize_particle_state():
    pos = np.random.rand(NUM_PARTICLES, 2) * np.array([WIDTH, HEIGHT])
    if RANDOM_INITIAL_VELOCITY:
        angles = np.random.rand(NUM_PARTICLES) * 2 * np.pi
        speeds = np.random.uniform(5, 50.0, size=NUM_PARTICLES)
        vel = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, np.newaxis]
    else:
        vel = np.random.randn(NUM_PARTICLES, 2)
    mass = np.random.uniform(0.5, 2.0, size=NUM_PARTICLES)
    return pos, vel, mass

def reset_simulation():
    global positions, velocities, masses, accelerations, selected_index
    positions, velocities, masses = initialize_particle_state()
    accelerations = np.zeros((NUM_PARTICLES, 2))
    selected_index = None

def mass_to_color(masses):
    norm = (masses - masses.min()) / (masses.max() - masses.min() + 1e-6)
    R = (255 * norm).astype(np.uint8)
    G = (255 * (1 - norm)).astype(np.uint8)
    B = (128 + 127 * (1 - norm)).astype(np.uint8)
    return list(zip(R, G, B))

def build_spatial_grid():
    spatial_grid.clear()
    N = len(positions)
    for i in range(N):
        x, y = positions[i]
        col = int(x // CELL_SIZE)
        row = int(y // CELL_SIZE)
        if 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS:
            spatial_grid[(col, row)].append(i)

def compute_accelerations_numpy():
    N = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2) + SOFTENING_EPS2
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    unit_vectors = diff / dist[:, :, np.newaxis]
    m_matrix = masses[:, np.newaxis] * masses[np.newaxis, :]
    force_mag = GRAVITATIONAL_CONSTANT * m_matrix / dist_sq
    forces = np.sum(force_mag[:, :, np.newaxis] * unit_vectors, axis=1)

    # globalne słońce
    if SUN_ENABLED:
        diff_s = SUN_POSITION - positions
        dist_sq_s = np.sum(diff_s**2, axis=1) + SOFTENING_EPS2
        dist_s = np.sqrt(dist_sq_s)
        unit_s = diff_s / dist_s[:, np.newaxis]
        force_mag_s = GRAVITATIONAL_CONSTANT * masses * SUN_MASS_GLOBAL / dist_sq_s
        forces += force_mag_s[:, np.newaxis] * unit_s

    # dodatkowe słońca
    for spos in extra_suns:
        diff_s = spos - positions
        dist_sq_s = np.sum(diff_s**2, axis=1) + SOFTENING_EPS2
        dist_s = np.sqrt(dist_sq_s)
        unit_s = diff_s / dist_s[:, np.newaxis]
        force_mag_s = GRAVITATIONAL_CONSTANT * masses * SUN_MASS_LOCAL / dist_sq_s
        forces += force_mag_s[:, np.newaxis] * unit_s

    acc = forces / masses[:, np.newaxis]
    if apply_global_gravity_y:
        acc += GLOBAL_GRAVITY_Y
    return acc

def handle_walls():
    N = len(positions)
    for i in range(N):
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        if positions[i, 0] < radius:
            positions[i, 0] = radius
            velocities[i, 0] = -WALL_RESTITUTION * velocities[i, 0]
        elif positions[i, 0] > WIDTH - radius:
            positions[i, 0] = WIDTH - radius
            velocities[i, 0] = -WALL_RESTITUTION * velocities[i, 0]
        if positions[i, 1] < radius:
            positions[i, 1] = radius
            velocities[i, 1] = -WALL_RESTITUTION * velocities[i, 1]
        elif positions[i, 1] > HEIGHT - radius:
            positions[i, 1] = HEIGHT - radius
            velocities[i, 1] = -WALL_RESTITUTION * velocities[i, 1]

def handle_sun_collision():
    N = len(positions)

    # globalne słońce
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

    # dodatkowe słońca
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
    positions = np.vstack([positions, np.array(pos)])
    angle = np.random.rand() * 2 * np.pi
    speed = np.random.uniform(1.0, 3.0)
    velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
    velocities = np.vstack([velocities, velocity])
    masses = np.append(masses, np.random.uniform(0.5, 2.0))
    accelerations = np.vstack([accelerations, np.zeros(2)])
    build_spatial_grid()
    accelerations = compute_accelerations_numpy()

def add_sun_at(pos):
    global extra_suns
    extra_suns.append(np.array(pos))

def remove_particle_at(pos):
    global positions, velocities, masses, accelerations
    N = len(positions)
    for i in range(N):
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        if np.linalg.norm(positions[i] - pos) < radius + 3:
            # usuń cząstkę ze wszystkich tablic
            positions = np.delete(positions, i, axis=0)
            velocities = np.delete(velocities, i, axis=0)
            masses = np.delete(masses, i)
            accelerations = np.delete(accelerations, i, axis=0)
            break  # zakończ po usunięciu jednej cząstki
    # przebuduj siatkę i policz akceleracje
    build_spatial_grid()
    accelerations = compute_accelerations_numpy()
    
    

def remove_sun_at(pos):
    global extra_suns
    for i, spos in enumerate(extra_suns):
        if np.linalg.norm(spos - pos) < sun_radius + 3:
            extra_suns.pop(i)
            break
            
            
            
def draw_particles():
    colors = mass_to_color(masses)
    N = len(positions)
    for i in range(N):
        x, y = positions[i]
        radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
        pygame.draw.circle(screen, colors[i], (int(x), int(y)), radius)

        # jeśli cząstka jest wybrana, pokaż wektor i etykietę
        if i == selected_index:
            # obwódka
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), radius + 2, 1)
            # wektor prędkości
            vx, vy = velocities[i]
            speed = np.linalg.norm([vx, vy])
            mass = masses[i]
            pygame.draw.line(screen, (255, 255, 0), (x, y), (x + vx * 10, y + vy * 10), 2)
            # etykieta
            label = font.render(f"v = {speed:.2f}, m = {mass:.2f}", True, (255, 255, 255))
            screen.blit(label, (x + 10, y - 20))

    # rysowanie globalnego słońca
    if SUN_ENABLED:
        pygame.draw.circle(screen, (255, 200, 0), SUN_POSITION.astype(int), sun_radius)

    # rysowanie dodatkowych słońc
    for spos in extra_suns:
        pygame.draw.circle(screen, (255, 150, 0), spos.astype(int), sun_radius)


    
    
# --- PĘTLA GŁÓWNA ---
reset_simulation()
build_spatial_grid()
accelerations = compute_accelerations_numpy()

running = True
while running:
    screen.fill((30, 30, 30))

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
            elif event.key == pygame.K_s:
                SUN_ENABLED = not SUN_ENABLED   # globalne słońce ON/OFF

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # lewy klik
                # sprawdź, czy kliknięto w cząstkę
                clicked_index = None
                for i in range(len(positions)):
                    radius = int(PARTICLE_RADIUS * np.sqrt(masses[i]))
                    if np.linalg.norm(positions[i] - event.pos) < radius + 3:
                        clicked_index = i
                        break

                if clicked_index is not None:
                    selected_index = clicked_index   # wybór cząstki
                else:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        add_sun_at(event.pos)        # Shift+klik → nowe słońce lokalne
                    else:
                        add_particle_at(event.pos)   # zwykła cząstka

            elif event.button == 3:  # prawy klik
                remove_particle_at(np.array(event.pos))
                remove_sun_at(np.array(event.pos))

    # --- Velocity Verlet ---
    positions += velocities * dt + 0.5 * accelerations * dt**2
    build_spatial_grid()
    new_acc = compute_accelerations_numpy()
    velocities += 0.5 * (accelerations + new_acc) * dt
    accelerations = new_acc
    step_count += 1
    sim_time += dt

    # --- Kolizje i odbicia ---
    handle_walls()
    handle_sun_collision()

    # --- Rysowanie cząstek i słońc ---
    draw_particles()

    # UI statusy
    grav_status = "ON" if apply_global_gravity_y else "OFF"
    grav_label = font.render(f"Global Gravity (G): {grav_status}", True, (255, 255, 255))
    screen.blit(grav_label, (10, 10))

    reset_label = font.render("Reset (R)", True, (255, 255, 255))
    screen.blit(reset_label, (10, 30))

    sun_status = "ON" if SUN_ENABLED else "OFF"
    sun_label = font.render(f"Global Sun (S): {sun_status}", True, (255, 255, 255))
    screen.blit(sun_label, (10, 50))

    step_label = font.render(f"Krok: {step_count}", True, (255, 255, 255))
    time_label = font.render(f"Czas: {sim_time:.2f} s", True, (255, 255, 255))
    screen.blit(step_label, (10, 70))
    screen.blit(time_label, (10, 90))

    # legenda sterowania
    legend = font.render("Lewy klik = cząstka | Shift+Lewy = słońce | Prawy = usuń", True, (200, 200, 200))
    screen.blit(legend, (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
