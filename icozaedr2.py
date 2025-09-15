import pygame
import math
import numpy as np

pygame.init()

# Ekran
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ikozaedr 3D z zoomem i cieniowaniem")

# Kolory
white = (255, 255, 255)
black = (0, 0, 0)
base_gray = (120, 120, 120)
line_color = (255, 255, 255)

# Parametry 3D
scale = 22
angle_x = angle_y = 0
fov = 256
viewer_dist = 50

# Wierzchołki ikozaedru
phi = (1 + math.sqrt(5)) / 2.0
vertices = np.array([
    (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
    ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
    ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1)
], dtype=float)

# Krawędzie
edges = [
    (0, 1), (0, 5), (0, 7), (0,10), (0,11),
    (1, 5), (1, 7), (1, 8), (1, 9),
    (2, 4), (2, 6), (2,10), (2,11),
    (3, 4), (3, 6), (3, 8), (3, 9),
    (4, 9), (4,11),
    (5, 9), (5,11),
    (6, 7), (6, 8), (6,10),
    (7, 8), (7,10),
    (8, 9),
    (10,11),
    (2, 3), (4, 5)
]

# Ściany (20 trójkątów)
faces = [
    (0, 1, 5), (0, 7, 1), (0,10, 7), (0,11,10), (0, 5,11),
    (1, 9, 5), (5, 9, 4), (5,11, 4), (11, 2, 4), (11,10, 2),
    (10, 6, 2), (10, 7, 6), (7, 8, 6), (7, 1, 8), (1, 9, 8),
    (3, 9, 8), (3, 4, 9), (3, 2, 4), (3, 6, 2), (3, 8, 6)
]

# Obrót punktu
def rotate_point(point, ax, ay):
    x, y, z = point
    xz = x * math.cos(ay) + z * math.sin(ay)
    zz = -x * math.sin(ay) + z * math.cos(ay)
    yz = y * math.cos(ax) - zz * math.sin(ax)
    zz = y * math.sin(ax) + zz * math.cos(ax)
    return np.array([xz, yz, zz])

# Rzutowanie
def project_point(p):
    x, y, z = p
    z += viewer_dist
    if z <= 0: return None
    factor = fov / z
    return (int(x * factor * scale + width / 2), int(y * factor * scale + height / 2))

# Sterowanie myszką
mouse_down = False
last_mouse_pos = (0, 0)
rotation_speed = 0.01

# Fonty
font = pygame.font.Font(None, 20)
font_face = pygame.font.Font(None, 18)

# Kierunek światła
light_dir = np.array([0.5, 1.0, -1.0])
light_dir /= np.linalg.norm(light_dir)

# Główna pętla
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_down = True
            last_mouse_pos = event.pos
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse_down = False
        if event.type == pygame.MOUSEMOTION and mouse_down:
            dx = event.pos[0] - last_mouse_pos[0]
            dy = event.pos[1] - last_mouse_pos[1]
            angle_y += dx * rotation_speed
            angle_x += dy * rotation_speed
            last_mouse_pos = event.pos
        if event.type == pygame.MOUSEWHEEL:
            scale += event.y * 2
            scale = max(5, min(100, scale))

    screen.fill(black)

    # Obrót i rzutowanie
    rotated = np.array([rotate_point(v, angle_x, angle_y) for v in vertices])
    projected = [project_point(p) for p in rotated]

    # Sortowanie ścian po głębokości
    face_depths = []
    for i, (a, b, c) in enumerate(faces):
        z_avg = (rotated[a][2] + rotated[b][2] + rotated[c][2]) / 3
        face_depths.append((z_avg, i))
    face_depths.sort()

    # Rysowanie ścian
    for _, idx in face_depths:
        a, b, c = faces[idx]
        p1, p2, p3 = rotated[a], rotated[b], rotated[c]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal) + 1e-9
        brightness = max(0.2, min(1.0, np.dot(normal, light_dir)))
        color = tuple(int(brightness * ch) for ch in base_gray)
        pts2d = [projected[a], projected[b], projected[c]]
        if all(p is not None for p in pts2d):
            pygame.draw.polygon(screen, color, pts2d)
            # Numer ściany
            centroid = (p1 + p2 + p3) / 3
            label_pos = project_point(centroid)
            if label_pos:
                label = font_face.render(str(idx), True, white)
                screen.blit(label, (label_pos[0] - 6, label_pos[1] - 6))

    # Krawędzie
    for a, b in edges:
        pa, pb = projected[a], projected[b]
        if pa and pb:
            pygame.draw.line(screen, line_color, pa, pb, 2)

    # Numery wierzchołków
    for i, pt in enumerate(projected):
        if pt:
            label = font.render(str(i), True, white)
            screen.blit(label, (pt[0] + 8, pt[1] + 8))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
