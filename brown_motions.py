import pygame
import random
import sys

# Ustawienia ekranu
WIDTH, HEIGHT = 800, 600
BG_COLOR = (25, 25, 25)

# Kolory cząstek
PARTICLE_COLOR = (255, 255, 255)  # Biały
MOLECULE_COLOR = (50, 50, 200)   # Niebieski

# Parametry symulacji
NUM_MOLECULES = 200
PARTICLE_RADIUS = 10
MOLECULE_RADIUS = 2
MAX_SPEED = 5
TRAIL_LENGTH = 100

# Inicjalizacja Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Symulacja Ruchu Browna")
clock = pygame.time.Clock()

class Particle:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.trail = []

    def draw(self):
        # Rysowanie śladu
        if len(self.trail) > 1:
            pygame.draw.lines(screen, self.color, False, self.trail, 1)
        # Rysowanie cząstki
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Molecule:
    def __init__(self, x, y, radius, color, speed):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.dx = random.uniform(-speed, speed)
        self.dy = random.uniform(-speed, speed)

    def move(self):
        self.x += self.dx
        self.y += self.dy
        
        # Odbicie od ścian
        if self.x <= 0 or self.x >= WIDTH:
            self.dx *= -1
        if self.y <= 0 or self.y >= HEIGHT:
            self.dy *= -1

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# Tworzenie cząstki i molekuł
particle = Particle(WIDTH // 2, HEIGHT // 2, PARTICLE_RADIUS, PARTICLE_COLOR)
molecules = [Molecule(random.randint(0, WIDTH), random.randint(0, HEIGHT), MOLECULE_RADIUS, MOLECULE_COLOR, MAX_SPEED) for _ in range(NUM_MOLECULES)]

# Główna pętla symulacji
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    screen.fill(BG_COLOR)
    
    # Przesuwanie molekuł
    for molecule in molecules:
        molecule.move()
        # Wykrywanie kolizji z cząstką Browna i zmiana jej położenia
        dx = particle.x - molecule.x
        dy = particle.y - molecule.y
        distance = (dx**2 + dy**2)**0.5
        if distance < particle.radius + molecule.radius:
            particle.x += molecule.dx
            particle.y += molecule.dy
            
    # Aktualizacja śladu
    particle.trail.append((particle.x, particle.y))
    if len(particle.trail) > TRAIL_LENGTH:
        particle.trail.pop(0)

    # Rysowanie obiektów
    for molecule in molecules:
        molecule.draw()
    particle.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
