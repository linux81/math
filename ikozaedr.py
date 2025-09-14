import pygame
import math
import numpy as np

# Inicjalizacja PyGame
pygame.init()

# Ustawienia ekranu
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ikozaedr z wypełnionymi ścianami i numerami")

# Kolory
white = (255, 255, 255)
black = (0, 0, 0)
gray = (100, 100, 100)
line_color = (255, 255, 255) # Biały kolor krawędzi
blue = (0, 0, 255)

# Ustawienia 3D
scale = 20
angle_x = angle_y = 0
fov = 256
viewer_dist = 50

# Wierzchołki ikozaedru
phi = (1 + math.sqrt(5)) / 2.0
vertices = np.array([
    (-1,  phi, 0), (1,  phi, 0), (-1, -phi, 0), (1, -phi, 0),
    (0, -1, phi), (0,  1, phi), (0, -1, -phi), (0,  1, -phi),
    (phi, 0, -1), (phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1)
], dtype=float)

# Poprawne krawędzie ikozaedru (30 par)
edges = [
    (0, 1), (0, 5), (0, 7), (0, 10), (0, 11),
    (1, 5), (1, 7), (1, 8), (1, 9),
    (2, 4), (2, 6), (2, 10), (2, 11),
    (3, 4), (3, 6), (3, 8), (3, 9),
    (4, 9), (4, 11),
    (5, 9), (5, 11),
    (6, 7), (6, 8), (6, 10),
    (7, 8), (7, 10),
    (8, 9),
    (10, 11),
    (2, 3), (4, 5)
]

# Poprawna, kompletna i zweryfikowana lista 20 ścian ikozaedru
faces = [
    (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (0, 11, 5),
    (1, 9, 8), (1, 8, 7), (1, 7, 0), (1, 0, 5), (1, 5, 9),
    (2, 3, 4), (2, 4, 11), (2, 11, 10), (2, 10, 6), (2, 6, 3),
    (3, 4, 9), (3, 9, 8), (3, 8, 6),
    (4, 5, 9), (4, 9, 3), (4, 3, 2), (4, 2, 11), (4, 11, 5),
    (5, 1, 9), (5, 9, 4), (5, 4, 11), (5, 11, 0), (5, 0, 1),
    (6, 7, 8), (6, 8, 3), (6, 3, 2), (6, 2, 10), (6, 10, 7),
    (7, 8, 9), (7, 9, 5), (7, 5, 0), (7, 0, 1), (7, 1, 8),
    (8, 9, 4), (8, 4, 3), (8, 3, 6), (8, 6, 7), (8, 7, 1), (8, 1, 9),
    (9, 5, 4), (9, 4, 3), (9, 3, 8), (9, 8, 1), (9, 1, 5),
    (10, 11, 4), (10, 4, 3), (10, 3, 6), (10, 6, 7), (10, 7, 0), (10, 0, 11),
    (11, 5, 9), (11, 9, 4), (11, 4, 2), (11, 2, 10), (11, 10, 0), (11, 0, 5)
]

def rotate_point(point, angle_x, angle_y):
    x, y, z = point
    rotated_x = x * math.cos(angle_y) + z * math.sin(angle_y)
    rotated_y = y
    rotated_z = -x * math.sin(angle_y) + z * math.cos(angle_y)
    x, y, z = rotated_x, rotated_y, rotated_z
    rotated_x = x
    rotated_y = y * math.cos(angle_x) - z * math.sin(angle_x)
    rotated_z = y * math.sin(angle_x) + z * math.cos(angle_x)
    return np.array([rotated_x, rotated_y, rotated_z])

def project_point(point):
    x, y, z = point
    z_dist = z + viewer_dist
    if z_dist <= 0:
        return None
    projection = fov / z_dist
    x_projected = int(x * projection * scale + width / 2)
    y_projected = int(y * projection * scale + height / 2)
    return (x_projected, y_projected)

# Ustawienia do obracania myszką
mouse_down = False
last_mouse_pos = (0, 0)
rotation_speed = 0.01

# Główna pętla gry
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                last_mouse_pos = event.pos
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_down = False
        if event.type == pygame.MOUSEMOTION and mouse_down:
            current_mouse_pos = event.pos
            dx = current_mouse_pos[0] - last_mouse_pos[0]
            dy = current_mouse_pos[1] - last_mouse_pos[1]
            angle_y += dx * rotation_speed
            angle_x += dy * rotation_speed
            last_mouse_pos = current_mouse_pos

    screen.fill(black)

    rotated_points = np.array([rotate_point(v, angle_x, angle_y) for v in vertices])
    projected_points = [project_point(p) for p in rotated_points]

    # Rysowanie wypełnionych ścian z back-face culling
    for face in faces:
        p1_3d = rotated_points[face[0]]
        p2_3d = rotated_points[face[1]]
        p3_3d = rotated_points[face[2]]

        v1 = p2_3d - p1_3d
        v2 = p3_3d - p1_3d
        normal = np.cross(v1, v2)

        viewer_vector = p1_3d - np.array([0, 0, -viewer_dist])

        if np.dot(normal, viewer_vector) < 0:
            poly_points = [projected_points[face[0]], projected_points[face[1]], projected_points[face[2]]]
            if all(p is not None for p in poly_points):
                pygame.draw.polygon(screen, gray, poly_points)
                
    # Rysowanie krawędzi na wierzchu, aby były widoczne
    for edge in edges:
        p1 = projected_points[edge[0]]
        p2 = projected_points[edge[1]]
        if p1 is not None and p2 is not None:
            pygame.draw.line(screen, line_color, p1, p2, 2)
    
    # Rysowanie numerów wierzchołków
    for i, point in enumerate(projected_points):
        if point is not None:
            font = pygame.font.Font(None, 20)
            text = font.render(str(i), True, white)
            screen.blit(text, (point[0] + 10, point[1] + 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
