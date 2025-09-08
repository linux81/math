# gray_scott_pygame_learn.py
# Interaktywny Gray-Scott (Pygame): nauka + eksperymenty
# Sterowanie:
#   Strzałki:        F (↑/↓), k (←/→)
#   Q/W:             Du (−/+)
#   A/S:             Dv (−/+)
#   +/-:             steps_per_frame (−/+)
#   1..5:            presety (plamy, labirynty, pierścienie, krople, plamy duże)
#   B:               warunki brzegowe (okresowe/Dirichlet/Neumann)
#   P:               pauza/wznowienie
#   R:               reset (impuls centralny)
#   N:               zasiej losowy szum
#   [, ]:            rozmiar pędzla (−/+)
#   G:               zapis PNG aktualnej ramki
#   C:               eksport CSV (U i V)
#   H:               pokaż/ukryj pomoc
# Mysz:
#   LPM przytrzymany: nanoszenie zakłóceń pędzlem (domyślnie zwiększa V)
#
# Wymagania: pip install numpy pygame
# (opcjonalnie) pip install matplotlib — ładniejsze palety kolorów

import sys
import time
import numpy as np
import pygame

try:
    import matplotlib.cm as cm
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# --- Parametry startowe ---
W, H = 360, 360               # rozmiar siatki (piksele = komórki)
Du, Dv = 0.16, 0.08
F, k = 0.060, 0.062
steps_per_frame = 8
brush_radius = 6
bc_modes = ["periodic", "dirichlet", "neumann"]
bc_idx = 0  # 0=periodic, 1=dirichlet, 2=neumann
palette_name = 'inferno' if HAVE_MPL else 'gray'
show_help = True
paused = False

# --- Presety (nazwy -> (Du, Dv, F, k)) ---
PRESETS = [
    ("Plamy (B-Zebra)", (0.16, 0.08, 0.060, 0.062)),
    ("Labirynty",       (0.14, 0.06, 0.035, 0.062)),
    ("Pierścienie",     (0.16, 0.08, 0.022, 0.051)),
    ("Krople",          (0.12, 0.05, 0.040, 0.060)),
    ("Plamy duże",      (0.20, 0.10, 0.060, 0.060)),
]

# --- Inicjalizacja Pygame ---
pygame.init()
pygame.display.set_caption("Gray-Scott Reaction-Diffusion — Pygame (nauka + eksperymenty)")
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 18)
big_font = pygame.font.SysFont(None, 22)

# --- Siatka U,V ---
U = np.ones((H, W), dtype=np.float32)
V = np.zeros((H, W), dtype=np.float32)

def seed_center():
    r = max(4, min(W, H)//10)
    cy, cx = H//2, W//2
    U[cy-r:cy+r, cx-r:cx+r] = 0.50
    V[cy-r:cy+r, cx-r:cx+r] = 0.25

def seed_noise(p=0.02):
    mask = np.random.rand(H, W) < p
    U[mask] = (0.50 + 0.5*np.random.rand(mask.sum())).astype(np.float32)
    V[mask] = (0.25 + 0.25*np.random.rand(mask.sum())).astype(np.float32)
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)

seed_center()

# --- Kolorowanie ---
def to_rgb(arr01):
    # arr01: [0,1] float32
    if HAVE_MPL and palette_name != 'gray':
        cmap = cm.get_cmap(palette_name)
        rgb = cmap(np.clip(arr01, 0, 1))[:, :, :3]  # RGBA -> RGB
        return (rgb * 255).astype(np.uint8)
    else:
        g = np.uint8(np.clip(arr01, 0, 1) * 255)
        return np.stack([g, g, g], axis=-1)

# --- Warunki brzegowe: laplasjan ---
def laplacian(Z):
    if bc_modes[bc_idx] == "periodic":
        return (
            -4.0 * Z
            + np.roll(Z, (0, -1), (0, 1))
            + np.roll(Z, (0, 1), (0, 1))
            + np.roll(Z, (-1, 0), (0, 1))
            + np.roll(Z, (1, 0), (0, 1))
        )
    elif bc_modes[bc_idx] == "dirichlet":
        Zp = np.pad(Z, 1, mode='constant', constant_values=0.0)
        return (
            -4.0 * Z
            + Zp[1:-1, 0:-2]
            + Zp[1:-1, 2:  ]
            + Zp[0:-2, 1:-1]
            + Zp[2:  , 1:-1]
        )
    else:  # neumann (zero-flux)
        Zp = np.pad(Z, 1, mode='edge')
        return (
            -4.0 * Z
            + Zp[1:-1, 0:-2]
            + Zp[1:-1, 2:  ]
            + Zp[0:-2, 1:-1]
            + Zp[2:  , 1:-1]
        )

# --- Krok symulacji ---
def step(n=1):
    global U, V
    for _ in range(n):
        Lu = laplacian(U)
        Lv = laplacian(V)
        UVV = U * (V * V)
        U += Du * Lu - UVV + F * (1.0 - U)
        V += Dv * Lv + UVV - (F + k) * V
        # Zabezpieczenia numeryczne
        np.clip(U, 0.0, 1.0, out=U)
        np.clip(V, 0.0, 1.0, out=V)
        U[np.isnan(U)] = 0.0
        V[np.isnan(V)] = 0.0
        U[np.isinf(U)] = 0.0
        V[np.isinf(V)] = 0.0

# --- Narzędzia: pędzel, zapisy ---
def apply_brush(mx, my, r):
    # Dodaj lokalne zakłócenie: zwiększ V i obniż U w okręgu
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - mx)**2 + (yy - my)**2 <= r*r
    V[mask] = np.clip(V[mask] + 0.35, 0.0, 1.0)
    U[mask] = np.clip(U[mask] - 0.35, 0.0, 1.0)

def save_png():
    ts = time.strftime("%Y%m%d_%H%M%S")
    arr = to_rgb(V)  # zapisujemy pole V
    surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
    pygame.image.save(surf, f"gray_scott_{ts}.png")
    print(f"[PNG] Zapisano gray_scott_{ts}.png")

def save_csv():
    ts = time.strftime("%Y%m%d_%H%M%S")
    np.savetxt(f"gray_scott_U_{ts}.csv", U, delimiter=',')
    np.savetxt(f"gray_scott_V_{ts}.csv", V, delimiter=',')
    print(f"[CSV] Zapisano gray_scott_U_{ts}.csv oraz gray_scott_V_{ts}.csv")

def apply_preset(idx):
    global Du, Dv, F, k
    name, (pDu, pDv, pF, pk) = PRESETS[idx]
    Du, Dv, F, k = pDu, pDv, pF, pk
    seed_center()
    print(f"[Preset] {name} -> Du={Du}, Dv={Dv}, F={F}, k={k}")

# --- Renderowanie overlay ---
def draw_overlay(fps, mouse_xy):
    lines = [
        f"F={F:.3f}  k={k:.3f}  Du={Du:.3f}  Dv={Dv:.3f}",
        f"BC={bc_modes[bc_idx]}  steps/frame={steps_per_frame}  brush={brush_radius}px",
        f"palette={palette_name}  FPS={fps:.1f}  paused={paused}",
    ]
    if mouse_xy is not None:
        x, y = mouse_xy
        if 0 <= x < W and 0 <= y < H:
            lines.append(f"(x,y)=({x},{y})  U={U[y,x]:.3f}  V={V[y,x]:.3f}")
    y0 = 6
    for line in lines:
        txt = font.render(line, True, (255, 255, 255))
        screen.blit(txt, (6, y0))
        y0 += 18
    if show_help:
        help_lines = [
            "Sterowanie:",
            "Strzałki: F (↑/↓), k (←/→) | Q/W: Du −/+ | A/S: Dv −/+",
            "+/-: steps/frame −/+ | B: warunki brzegowe | 1..5: presety",
            "P: pauza | R: reset | N: szum | [, ]: pędzel −/+",
            "G: zapis PNG | C: eksport CSV | H: pomoc on/off",
            "Mysz LPM: nanoszenie zakłóceń (pędzel)"
        ]
        pad = 6
        box_w = max(font.size(h)[0] for h in help_lines) + 2*pad
        box_h = len(help_lines)*18 + 2*pad
        s = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 140))
        screen.blit(s, (W - box_w - 6, 6))
        y = 6 + pad
        x = W - box_w - 6 + pad
        for i, h in enumerate(help_lines):
            fnt = big_font if i == 0 else font
            col = (255, 255, 0) if i == 0 else (230, 230, 230)
            screen.blit(fnt.render(h, True, col), (x, y))
            y += 18

# --- Główna pętla ---
last_time = time.time()
fps = 0.0
mouse_down = False

while True:
    # Zdarzenia
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_down = False

        elif event.type == pygame.KEYDOWN:
            # Presety 1..5
            if pygame.K_1 <= event.key <= pygame.K_5:
                idx = event.key - pygame.K_1
                if 0 <= idx < len(PRESETS):
                    apply_preset(idx)

            elif event.key == pygame.K_b:
                bc_idx = (bc_idx + 1) % len(bc_modes)

            elif event.key == pygame.K_p:
                paused = not paused

            elif event.key == pygame.K_r:
                seed_center()

            elif event.key == pygame.K_n:
                seed_noise()

            elif event.key == pygame.K_g:
                save_png()

            elif event.key == pygame.K_c:
                save_csv()

            elif event.key == pygame.K_h:
                show_help = not show_help

            elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN,
                               pygame.K_q, pygame.K_w, pygame.K_a, pygame.K_s,
                               pygame.K_PLUS, pygame.K_EQUALS, pygame.K_MINUS, pygame.K_KP_PLUS, pygame.K_KP_MINUS,
                               pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                # Regulacje parametrów
                if event.key == pygame.K_UP:     F += 0.001
                if event.key == pygame.K_DOWN:   F -= 0.001
                if event.key == pygame.K_RIGHT:  k += 0.001
                if event.key == pygame.K_LEFT:   k -= 0.001
                if event.key == pygame.K_q:      Du -= 0.005
                if event.key == pygame.K_w:      Du += 0.005
                if event.key == pygame.K_a:      Dv -= 0.005
                if event.key == pygame.K_s:      Dv += 0.005
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS): steps_per_frame += 1
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):               steps_per_frame -= 1
                if event.key == pygame.K_LEFTBRACKET:  brush_radius = max(1, brush_radius - 1)
                if event.key == pygame.K_RIGHTBRACKET: brush_radius = min(64, brush_radius + 1)
                # Ograniczenia bezpieczne
                F = float(np.clip(F, 0.0, 0.09))
                k = float(np.clip(k, 0.0, 0.09))
                Du = float(np.clip(Du, 0.0, 0.50))
                Dv = float(np.clip(Dv, 0.0, 0.50))
                steps_per_frame = int(np.clip(steps_per_frame, 1, 100))

    # Pędzel myszy
    mx, my = pygame.mouse.get_pos()
    if mouse_down:
        apply_brush(mx, my, brush_radius)

    # Symulacja
    if not paused:
        step(steps_per_frame)

    # Render
    img_rgb = to_rgb(V)  # wizualizujemy V
    surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
    screen.blit(surf, (0, 0))

    # Overlay
    now = time.time()
    dt = now - last_time
    if dt > 0:
        fps = 0.9*fps + 0.1*(1.0/dt)
    last_time = now
    draw_overlay(fps, (mx, my))

    pygame.display.flip()
    clock.tick(60)
