import pygame
import random

# Inicjalizacja Pygame
pygame.init()

# Ustawienia ekranu
width, height = 800, 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Trójkąt Sierpińskiego")

# Kolory
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLOR_PALETTE = [
    (153, 204, 255),  # Jasny niebieski
    (102, 178, 255),  # Średni niebieski
    (51, 153, 255),   # Głęboki niebieski
    (0, 128, 255),    # Ciemnoniebieski
    (0, 102, 204),    # Granatowy
]

# Poziom rekurencji (jak wiele "podziałów" ma mieć trójkąt)
RECURSION_LEVEL = 7

# Wierzchołki początkowego trójkąta
# Zauważ, że dostosowujemy je do wymiarów okna
points = [(width // 2, 50), (50, height - 50), (width - 50, height - 50)]

def draw_sierpinski_triangle(points, level):
    """
    Rysuje trójkąt Sierpińskiego rekurencyjnie.

    Args:
        points (list): Lista zawierająca 3 krotki, które reprezentują
                       współrzędne wierzchołków trójkąta.
        level (int): Aktualny poziom rekurencji.
    """
    if level == 0:
        # Kiedy osiągniemy najniższy poziom, rysujemy mały trójkąt
        # Używamy koloru z palety w zależności od poziomu
        color = COLOR_PALETTE[min(RECURSION_LEVEL - 1, len(COLOR_PALETTE) - 1)]
        pygame.draw.polygon(screen, color, points)
        return

    # Obliczanie punktów środkowych boków
    mid_a = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
    mid_b = ((points[1][0] + points[2][0]) // 2, (points[1][1] + points[2][1]) // 2)
    mid_c = ((points[2][0] + points[0][0]) // 2, (points[2][1] + points[0][1]) // 2)

    # Wywołanie rekurencyjne dla trzech mniejszych trójkątów
    draw_sierpinski_triangle([points[0], mid_a, mid_c], level - 1) # Górny trójkąt
    draw_sierpinski_triangle([mid_a, points[1], mid_b], level - 1) # Lewy dolny trójkąt
    draw_sierpinski_triangle([mid_c, mid_b, points[2]], level - 1) # Prawy dolny trójkąt


def main():
    """
    Główna pętla programu.
    """
    running = True
    while running:
        # Obsługa zdarzeń, np. zamknięcie okna
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Wypełnienie tła
        screen.fill(BLACK)

        # Rysowanie trójkąta Sierpińskiego
        draw_sierpinski_triangle(points, RECURSION_LEVEL)

        # Odświeżenie ekranu
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
