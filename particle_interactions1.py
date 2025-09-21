import pygame
import random
import math

# --- Ustawienia symulacji ---
WIDTH, HEIGHT = 800, 600
PARTICLE_COUNT = 150
PARTICLE_RADIUS = 3
PARTICLE_MASS = 1.0
GRAVITY_STRENGTH = 0.5  # Siła przyciągania między cząsteczkami
REPULSION_STRENGTH = 1000  # Siła odpychania przy kolizji
REPULSION_DISTANCE = 20  # Odległość, poniżej której występuje silne odpychanie
TIME_STEP = 0.1  # Krok czasowy symulacji (mniejsza wartość = większa stabilność)
BACKGROUND_COLOR = (0, 0, 0)
PARTICLE_COLOR = (255, 255, 255)

# --- Klasa Cząsteczki ---
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1) * 0.5  # Losowa prędkość początkowa
        self.vy = random.uniform(-1, 1) * 0.5
        self.radius = PARTICLE_RADIUS
        self.mass = PARTICLE_MASS

    def update(self):
        # Aktualizacja pozycji na podstawie prędkości
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP

    def draw(self, screen):
        # Rysowanie cząsteczki na ekranie
        pygame.draw.circle(screen, PARTICLE_COLOR, (int(self.x), int(self.y)), self.radius)

# --- Inicjalizacja Pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Symulacja Oddziaływań Cząsteczek")
clock = pygame.time.Clock()

# Tworzenie cząsteczek
particles = [Particle(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(PARTICLE_COUNT)]

# --- Główna pętla symulacji ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Obliczanie sił i aktualizacja cząsteczek ---
    for i in range(PARTICLE_COUNT):
        p1 = particles[i]
        
        # Obliczanie siły odpychania od ścian
        if p1.x <= p1.radius or p1.x >= WIDTH - p1.radius:
            p1.vx *= -1
        if p1.y <= p1.radius or p1.y >= HEIGHT - p1.radius:
            p1.vy *= -1

        # Obliczanie oddziaływań między cząsteczkami
        for j in range(i + 1, PARTICLE_COUNT):
            p2 = particles[j]
            
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            distance = math.sqrt(dx**2 + dy**2)
            
            # Zapobieganie błędom dzielenia przez zero i zbyt dużym siłom
            if distance < 1:
                distance = 1

            # Siła odpychania (jeśli cząsteczki są blisko)
            if distance < REPULSION_DISTANCE:
                force = -REPULSION_STRENGTH / (distance**2)
                
                # Dodajemy losowy element, żeby uniknąć symetrycznych wzorców
                force += random.uniform(-0.1, 0.1) 
            # Siła przyciągania (jeśli cząsteczki są dalej)
            else:
                force = GRAVITY_STRENGTH / (distance**2)

            # Obliczanie składowych siły (x i y)
            fx = force * dx / distance
            fy = force * dy / distance
            
            # Zastosowanie siły na obu cząsteczkach (zasada akcji i reakcji)
            p1.vx += fx / p1.mass * TIME_STEP
            p1.vy += fy / p1.mass * TIME_STEP
            p2.vx -= fx / p2.mass * TIME_STEP
            p2.vy -= fy / p2.mass * TIME_STEP

    # Aktualizacja pozycji wszystkich cząsteczek
    for p in particles:
        p.update()

    # --- Rysowanie na ekranie ---
    screen.fill(BACKGROUND_COLOR)
    for p in particles:
        p.draw(screen)
    pygame.display.flip()
    
    # Ograniczenie liczby klatek na sekundę
    clock.tick(60)

pygame.quit()
