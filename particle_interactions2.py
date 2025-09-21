import pygame
import random
import math
import csv

# --- Ustawienia symulacji ---
WIDTH, HEIGHT = 800, 600
PARTICLE_COUNT = 300
PARTICLE_MASS = 1.0
PARTICLE_RADIUS = 3
GRAVITY_STRENGTH = 5
REPULSION_STRENGTH = 1000
REPULSION_DISTANCE = 20
TIME_STEP = 0.1
BACKGROUND_COLOR = (0, 0, 0)

# --- Opcje dodatkowe ---
COLOR_BY_SPEED = True
SHOW_POTENTIAL_FIELD = False
SHOW_TOTAL_ENERGY = False
SHOW_SPEED_HISTOGRAM = False
RANDOM_MASS = True
MASS_MIN = 0.5
MASS_MAX = 2.0
SCALE_RADIUS_BY_MASS = True

# --- Klasa Cząsteczki ---
class Particle:
    TYPES = {
        "A": {"color": (255, 100, 100)},
        "B": {"color": (100, 255, 100)},
        "C": {"color": (100, 100, 255)},
    }

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1) * 0.5
        self.vy = random.uniform(-1, 1) * 0.5
        self.type = random.choice(list(Particle.TYPES.keys()))
        self.color = Particle.TYPES[self.type]["color"]
        self.mass = random.uniform(MASS_MIN, MASS_MAX) if RANDOM_MASS else PARTICLE_MASS
        base_radius = PARTICLE_RADIUS
        self.radius = int(base_radius * self.mass) if SCALE_RADIUS_BY_MASS else base_radius

    def update(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP

    def draw(self, screen):
        if COLOR_BY_SPEED:
            speed = math.sqrt(self.vx**2 + self.vy**2)
            intensity = min(255, int(speed * 100))
            color = tuple(min(255, int(c * (intensity / 255))) for c in self.color)
        else:
            color = self.color
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)

# --- Pole potencjału ---
def draw_potential_field(screen, particles, resolution=20):
    for x in range(0, WIDTH, resolution):
        for y in range(0, HEIGHT, resolution):
            potential = 0
            for p in particles:
                dx = p.x - x
                dy = p.y - y
                dist_sq = dx**2 + dy**2
                if dist_sq > 1:
                    potential += -GRAVITY_STRENGTH * p.mass / dist_sq
            intensity = max(0, min(255, int(-potential * 100)))
            color = (intensity, intensity, intensity)
            pygame.draw.rect(screen, color, (x, y, resolution, resolution))

# --- Energia układu ---
def compute_total_energy(particles):
    kinetic = sum(0.5 * p.mass * (p.vx**2 + p.vy**2) for p in particles)
    potential = 0
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            dx = particles[j].x - particles[i].x
            dy = particles[j].y - particles[i].y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 1:
                potential += -GRAVITY_STRENGTH * particles[i].mass * particles[j].mass / dist
    return kinetic, potential

# --- Histogram prędkości ---
def draw_speed_histogram(screen, particles, bins=10, width=200, height=100, pos=(WIDTH - 210, 10)):
    speeds = [math.sqrt(p.vx**2 + p.vy**2) for p in particles]
    max_speed = max(speeds) if speeds else 1
    bin_counts = [0] * bins
    for s in speeds:
        index = min(bins - 1, int(s / max_speed * bins))
        bin_counts[index] += 1
    max_count = max(bin_counts)
    for i, count in enumerate(bin_counts):
        bar_height = int(count / max_count * height)
        x = pos[0] + i * (width // bins)
        y = pos[1] + height - bar_height
        pygame.draw.rect(screen, (0, 255, 0), (x, y, width // bins - 2, bar_height))
    font = pygame.font.SysFont(None, 20)
    screen.blit(font.render("Histogram prędkości", True, (255, 255, 255)), (pos[0], pos[1] + height + 5))

# --- Zapis do CSV ---
def save_particle_data(particles, filename="particles.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "vx", "vy", "mass", "type"])
        for p in particles:
            writer.writerow([p.x, p.y, p.vx, p.vy, p.mass, p.type])

# --- Inicjalizacja Pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Symulacja Oddziaływań Cząsteczek")
clock = pygame.time.Clock()

particles = [Particle(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(PARTICLE_COUNT)]

# --- Główna pętla ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                GRAVITY_STRENGTH += 0.1
            elif event.key == pygame.K_DOWN:
                GRAVITY_STRENGTH = max(0, GRAVITY_STRENGTH - 0.1)
            elif event.key == pygame.K_RIGHT:
                REPULSION_STRENGTH += 100
            elif event.key == pygame.K_LEFT:
                REPULSION_STRENGTH = max(0, REPULSION_STRENGTH - 100)
            elif event.key == pygame.K_s:
                save_particle_data(particles)

    for i in range(PARTICLE_COUNT):
        p1 = particles[i]
        if p1.x <= p1.radius or p1.x >= WIDTH - p1.radius:
            p1.vx *= -1
        if p1.y <= p1.radius or p1.y >= HEIGHT - p1.radius:
            p1.vy *= -1

        for j in range(i + 1, PARTICLE_COUNT):
            p2 = particles[j]
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 1:
                distance = 1
            if distance < REPULSION_DISTANCE:
                force = -REPULSION_STRENGTH / (distance**2)
                force += random.uniform(-0.1, 0.1)
            else:
                force = GRAVITY_STRENGTH / (distance**2)
            fx = force * dx / distance
            fy = force * dy / distance
            p1.vx += fx / p1.mass * TIME_STEP
            p1.vy += fy / p1.mass * TIME_STEP
            p2.vx -= fx / p2.mass * TIME_STEP
            p2.vy -= fy / p2.mass * TIME_STEP

    for p in particles:
        p.update()

    screen.fill(BACKGROUND_COLOR)

    if SHOW_POTENTIAL_FIELD:
        draw_potential_field(screen, particles)

    if SHOW_TOTAL_ENERGY:
        kinetic, potential = compute_total_energy(particles)
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"E_kin={kinetic:.1f}  E_pot={potential:.1f}  E_tot={kinetic+potential:.1f}", True, (255, 255, 0))
        screen.blit(text, (10, 10))

    if SHOW_SPEED_HISTOGRAM:
        draw_speed_histogram(screen, particles)

    font = pygame.font.SysFont(None, 20)
    param_text = font.render(f"G={GRAVITY_STRENGTH:.2f}  R={REPULSION_STRENGTH:.0f}", True, (200, 200, 255))
    screen.blit(param_text, (10, 35))

    for p in particles:
        p.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
