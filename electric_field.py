import pygame
import math
import random

# Inicjalizacja Pygame
pygame.init()

# Ustawienia ekranu
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Symulacja pola elektrycznego 2D")

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Stała elektrostatyczna
K_SIM = 100000

# Klasa cząstki
class Particle:
    def __init__(self, x, y, mass, charge, color):
        self.x = x
        self.y = y
        self.mass = mass
        self.charge = charge
        self.color = color
        self.radius = 5
        self.vx = 0
        self.vy = 0
        self.history = []

    def apply_force(self, fx, fy, dt):
        """Oblicza przyspieszenie i aktualizuje prędkość."""
        ax = fx / self.mass
        ay = fy / self.mass
        self.vx += ax * dt
        self.vy += ay * dt

    def update_position(self, dt):
        """Aktualizuje pozycję i zapisuje historię."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.history.append((self.x, self.y))
        if len(self.history) > 200:
            self.history.pop(0)

    def draw(self):
        """Rysuje cząstkę i jej trajektorię."""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        if len(self.history) > 1:
            pygame.draw.lines(screen, self.color, False, self.history, 1)

# Klasa źródła pola elektrycznego
class ElectricFieldSource:
    def __init__(self, x, y, charge, color):
        self.x = x
        self.y = y
        self.charge = charge
        self.color = color
        self.radius = 10
        self.is_dragging = False

    def draw(self):
        """Rysuje źródło."""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.radius, 2)
        font = pygame.font.Font(None, 24)
        text = font.render("+" if self.charge > 0 else "-", True, BLACK)
        text_rect = text.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(text, text_rect)
    
    def contains_point(self, pos):
        """Sprawdza, czy punkt (np. kursor myszy) znajduje się wewnątrz źródła."""
        return math.sqrt((self.x - pos[0])**2 + (self.y - pos[1])**2) < self.radius

# Funkcja obliczająca siłę Coulomba
def calculate_coulomb_force(p1, p2):
    """Oblicza wektor siły między dwoma naładowanymi obiektami."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    distance_sq = dx**2 + dy**2
    if distance_sq < 25:
        distance_sq = 25
    
    distance = math.sqrt(distance_sq)
    
    # Prawo Coulomba
    force = K_SIM * (p1.charge * p2.charge) / distance_sq
    
    fx = force * (dx / distance)
    fy = force * (dy / distance)
    return fx, fy

# Tworzenie obiektów
particles = []

# Źródła pola elektrycznego
sources = [
    ElectricFieldSource(WIDTH // 2 - 100, HEIGHT // 2, charge=5, color=RED),
    ElectricFieldSource(WIDTH // 2 + 100, HEIGHT // 2, charge=-5, color=BLUE)
]

# Główna pętla symulacji
running = True
clock = pygame.time.Clock()
dt = 0.05  # Krok czasowy

while running:
    # Obsługa zdarzeń
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Resetowanie symulacji
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                particles.clear()
        
        # Obsługa zdarzeń myszy - dodawanie cząstek
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            keys = pygame.key.get_pressed()

            # Dodawanie cząstki (LPM)
            if event.button == 1:
                # Cząstka ciężka (proton) - Shift + LPM
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    new_charge = random.choice([-1, 1])
                    new_color = RED if new_charge > 0 else BLUE
                    particles.append(Particle(mouse_x, mouse_y, mass=10, charge=new_charge, color=new_color))
                # Cząstka lekka (elektron) - LPM
                else:
                    new_charge = random.choice([-1, 1])
                    new_color = RED if new_charge > 0 else BLUE
                    particles.append(Particle(mouse_x, mouse_y, mass=1, charge=new_charge, color=new_color))

            # Przesuwanie źródeł (PPM)
            if event.button == 3:  # 3 to prawy przycisk myszy
                for source in sources:
                    if source.contains_point(event.pos):
                        source.is_dragging = True
                        break
        
        # Zakończenie przeciągania
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:
                for source in sources:
                    source.is_dragging = False

    # Przeciąganie źródeł
    for source in sources:
        if source.is_dragging:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            source.x = mouse_x
            source.y = mouse_y
    
    # Czyszczenie ekranu
    screen.fill(WHITE)
    
    # Obliczanie sił i aktualizacja pozycji
    for particle in particles:
        total_fx = 0
        total_fy = 0
        
        # Siła od źródeł
        for source in sources:
            fx, fy = calculate_coulomb_force(particle, source)
            total_fx += fx
            total_fy += fy
            
        # Siła od innych cząstek (interakcje między nimi)
        for other_particle in particles:
            if particle is not other_particle:
                fx, fy = calculate_coulomb_force(particle, other_particle)
                total_fx += fx
                total_fy += fy

        particle.apply_force(total_fx, total_fy, dt)
        particle.update_position(dt)
        
    # Rysowanie
    for source in sources:
        source.draw()
    for particle in particles:
        particle.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
