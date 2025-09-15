import pygame
import math
import numpy as np

pygame.init()
width, height = 900, 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Bryły platońskie 3D — obrót, zoom, przełączanie (1–5)")

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BASE_GRAY = (120, 120, 120)
EDGE_COLOR = (240, 240, 240)

# Parametry kamery i projekcji
scale = 26
angle_x = 0.0
angle_y = 0.0
fov = 256
viewer_dist = 60

# Mysz
mouse_down = False
last_mouse_pos = (0, 0)
rotation_speed = 0.01

# Fonty
font_v = pygame.font.Font(None, 20)
font_f = pygame.font.Font(None, 18)
font_ui = pygame.font.Font(None, 24)

# Kierunek światła (stały, z góry i lekko z przodu)
light_dir = np.array([0.5, 1.0, -1.0], dtype=float)
light_dir /= np.linalg.norm(light_dir)

def rotate_point(p, ax, ay):
    x, y, z = p
    # Y-rot
    cx, sx = math.cos(ay), math.sin(ay)
    x2 = x * cx + z * sx
    z2 = -x * sx + z * cx
    # X-rot
    cy, sy = math.cos(ax), math.sin(ax)
    y2 = y * cy - z2 * sy
    z3 = y * sy + z2 * cy
    return np.array([x2, y2, z3], dtype=float)

def project_point(p):
    x, y, z = p
    z_cam = z + viewer_dist
    if z_cam <= 0:
        return None
    k = fov / z_cam
    return (int(x * k * scale + width / 2), int(y * k * scale + height / 2))

def normalize_vertices(vertices):
    # Ujednolica wielkość brył: maksymalna odległość od środka = 1
    norms = np.linalg.norm(vertices, axis=1)
    max_norm = max(1e-9, norms.max())
    return vertices / max_norm

def build_edges_from_faces(faces):
    # Wyznacza unikalne krawędzie z listy ścian (poligony)
    edges = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            edge = (min(a, b), max(a, b))
            edges.add(edge)
    return sorted(edges)

def triangulate_face(face):
    # Triangulacja wachlarzem: (v0, v1, v2), (v0, v2, v3), ...
    if len(face) == 3:
        return [tuple(face)]
    tris = []
    for i in range(1, len(face) - 1):
        tris.append((face[0], face[i], face[i + 1]))
    return tris

# Definicje brył (wierzchołki i ściany jako poligony)
def solid_tetrahedron():
    v = np.array([
        [ 1,  1,  1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
    ], dtype=float)
    f = [
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ]
    return v, f

def solid_cube():
    v = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], dtype=float)
    # Ściany kwadratowe (będą triangulowane)
    f = [
        [0, 1, 2, 3],   # tył
        [4, 5, 6, 7],   # przód
        [0, 4, 5, 1],   # dół
        [3, 7, 6, 2],   # góra
        [0, 3, 7, 4],   # lewa
        [1, 5, 6, 2],   # prawa
    ]
    return v, f

def solid_octahedron():
    v = np.array([
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
    ], dtype=float)
    f = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [0, 5, 2], [2, 5, 1], [1, 5, 3], [3, 5, 0],
    ]
    return v, f

def solid_icosahedron():
    phi = (1 + math.sqrt(5)) / 2.0
    v = np.array([
        (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1)
    ], dtype=float)
    f = [
        [0, 1, 5], [0, 7, 1], [0,10, 7], [0,11,10], [0, 5,11],
        [1, 9, 5], [5, 9, 4], [5,11, 4], [11, 2, 4], [11,10, 2],
        [10, 6, 2], [10, 7, 6], [7, 8, 6], [7, 1, 8], [1, 9, 8],
        [3, 9, 8], [3, 4, 9], [3, 2, 4], [3, 6, 2], [3, 8, 6],
    ]
    return v, f

def solid_dodecahedron_dual():
    # Budujemy dodekaedr jako dual ikozaedru:
    # - wierzchołki: środki ścian ikozaedru (20)
    # - ściany: dla każdego wierzchołka ikozaedru z 5 sąsiednich ścian tworzymy pentagon
    ico_v, ico_f = solid_icosahedron()
    ico_v = normalize_vertices(ico_v)

    # Środki trójkątów -> 20 wierzchołków dodekaedru
    centers = []
    for a, b, c in ico_f:
        centers.append((ico_v[a] + ico_v[b] + ico_v[c]) / 3.0)
    centers = np.array(centers, dtype=float)
    centers = normalize_vertices(centers)

    # Mapowanie: dla każdego wierzchołka ikozaedru lista twarzy (indeksów trójkątów), które go zawierają
    faces_by_vertex = [[] for _ in range(len(ico_v))]
    for fi, tri in enumerate(ico_f):
        for v_idx in tri:
            faces_by_vertex[v_idx].append(fi)

    # Porządkowanie pięciu środków wokół każdego wierzchołka — sortowanie kątowe w lokalnej bazie
    dodec_faces = []
    for v_idx, face_indices in enumerate(faces_by_vertex):
        vpos = ico_v[v_idx]
        n = vpos / (np.linalg.norm(vpos) + 1e-9)
        # wybór wektora pomocniczego niezrównoległego do n
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(up, n)) > 0.9:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        u = np.cross(up, n)
        u /= (np.linalg.norm(u) + 1e-9)
        v = np.cross(n, u)

        # Parowanie: (kąt, index_środka)
        ang_list = []
        for ci in face_indices:
            vec = centers[ci] - vpos
            x = np.dot(vec, u)
            y = np.dot(vec, v)
            ang = math.atan2(y, x)
            ang_list.append((ang, ci))

        ang_list.sort()
        ordered = [ci for _, ci in ang_list]  # pentagon wokół v_idx
        dodec_faces.append(ordered)

    return centers, dodec_faces

def get_platonic_solid(name):
    if name == "tetrahedron": return solid_tetrahedron()
    if name == "cube": return solid_cube()
    if name == "octahedron": return solid_octahedron()
    if name == "dodecahedron": return solid_dodecahedron_dual()
    if name == "icosahedron": return solid_icosahedron()
    return np.zeros((0, 3)), []

# Startowa bryła
solid_name = "icosahedron"
vertices, faces = get_platonic_solid(solid_name)
vertices = normalize_vertices(vertices)
edges = build_edges_from_faces(faces)

clock = pygame.time.Clock()
running = True

def draw_scene(vertices, faces, edges):
    # Obrót i rzutowanie punktów
    rotated = np.array([rotate_point(v, angle_x, angle_y) for v in vertices])
    projected = [project_point(p) for p in rotated]

    # Sortowanie ścian po średnim Z (Painter)
    face_depths = []
    for i, face in enumerate(faces):
        z_avg = np.mean([rotated[idx][2] for idx in face])
        face_depths.append((z_avg, i))
    face_depths.sort()  # od dalszych do bliższych

    cam_pos = np.array([0.0, 0.0, -viewer_dist], dtype=float)

    # Rysowanie wypełnionych ścian z cieniowaniem i cullingiem
    for _, fi in face_depths:
        face = faces[fi]
        if len(face) < 3:
            continue

        # Normalna liczona z pierwszych trzech wierzchołków (w 3D po obrocie)
        p1 = rotated[face[0]]
        p2 = rotated[face[1]]
        p3 = rotated[face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            continue
        n = n / n_norm

        # Back-face culling względem kamery
        centroid = np.mean([rotated[idx] for idx in face], axis=0)
        view_vec = cam_pos - centroid
        if np.dot(n, view_vec) <= 0:
            continue

        # Cieniowanie: stałe światło kierunkowe
        brightness = max(0.2, min(1.0, float(np.dot(n, light_dir))))
        color = tuple(int(brightness * c) for c in BASE_GRAY)

        poly_pts = [projected[idx] for idx in face]
        if any(pt is None for pt in poly_pts):
            continue

        pygame.draw.polygon(screen, color, poly_pts)

        # Etykieta ściany (indeks)
        centroid_2d = project_point(centroid)
        if centroid_2d is not None:
            label = font_f.render(str(fi), True, WHITE)
            screen.blit(label, (centroid_2d[0] - 6, centroid_2d[1] - 6))

    # Krawędzie na wierzchu
    for a, b in edges:
        pa, pb = projected[a], projected[b]
        if pa is not None and pb is not None:
            pygame.draw.line(screen, EDGE_COLOR, pa, pb, 2)

    # Numery wierzchołków
    for i, pt in enumerate(projected):
        if pt is not None:
            label = font_v.render(str(i), True, WHITE)
            screen.blit(label, (pt[0] + 8, pt[1] + 8))

def reset_orientation():
    global angle_x, angle_y
    angle_x = 0.0
    angle_y = 0.0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_down = True
            last_mouse_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse_down = False

        elif event.type == pygame.MOUSEMOTION and mouse_down:
            dx = event.pos[0] - last_mouse_pos[0]
            dy = event.pos[1] - last_mouse_pos[1]
            angle_y += dx * rotation_speed
            angle_x += dy * rotation_speed
            last_mouse_pos = event.pos

        elif event.type == pygame.MOUSEWHEEL:
            scale += event.y * 2
            scale = max(6, min(140, scale))

        elif event.type == pygame.KEYDOWN:
            changed = False
            if event.key == pygame.K_1:
                solid_name = "tetrahedron"; changed = True
            elif event.key == pygame.K_2:
                solid_name = "cube"; changed = True
            elif event.key == pygame.K_3:
                solid_name = "octahedron"; changed = True
            elif event.key == pygame.K_4:
                solid_name = "dodecahedron"; changed = True
            elif event.key == pygame.K_5:
                solid_name = "icosahedron"; changed = True
            elif event.key == pygame.K_r:
                reset_orientation()
            if changed:
                vertices, faces = get_platonic_solid(solid_name)
                vertices = normalize_vertices(vertices)
                edges = build_edges_from_faces(faces)
                reset_orientation()

    screen.fill(BLACK)
    draw_scene(vertices, faces, edges)

    # UI podpowiedzi
    ui_lines = [
        "Sterowanie: LPM — obrót, kółko — zoom, R — reset",
        "1: Czworościan  2: Sześcian  3: Ośmiościan  4: Dwudziestościan  5: Ikozaedr",
        f"Aktualna bryła: {solid_name}",
    ]
    y = 10
    for line in ui_lines:
        surf = font_ui.render(line, True, WHITE)
        screen.blit(surf, (10, y))
        y += 24

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
