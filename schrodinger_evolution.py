"""
Time-Dependent Schrödinger Evolution
=====================================
Interactive Pygame visualization of a quantum wavepacket evolving in
various potentials. See Re[ψ], Im[ψ], and |ψ|² in real time.

Demonstrates: tunneling, reflection, dispersion, and interference.
"""

import sys
import math
import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1400, 900
FPS = 60

BG_COLOR = (13, 17, 23)
PANEL_BG = (22, 27, 34)
BORDER_COLOR = (48, 54, 61)
TEXT_COLOR = (201, 209, 217)
DIM_TEXT = (125, 133, 144)
TITLE_COLOR = (240, 246, 252)

RE_COLOR = (99, 179, 237)       # blue for Re[ψ]
IM_COLOR = (236, 121, 154)      # pink for Im[ψ]
PROB_COLOR = (167, 139, 250)    # violet for |ψ|²
POTENTIAL_COLOR = (250, 204, 21)
POTENTIAL_FILL = (250, 204, 21, 50)

NX = 1200
X_MIN, X_MAX = -30.0, 30.0
DX = (X_MAX - X_MIN) / NX
X = np.linspace(X_MIN, X_MAX, NX)

SCENARIOS = [
    "Free Propagation",
    "Barrier Tunneling",
    "Double Slit Interference",
    "Harmonic Oscillator",
    "Step Potential",
]

# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------

def potential_free(x):
    return np.zeros_like(x)

def potential_barrier(x):
    V = np.zeros_like(x)
    V[(x > 2.0) & (x < 3.5)] = 0.8
    return V

def potential_double_slit(x):
    V = np.zeros_like(x)
    wall = (x > 4.0) & (x < 5.0)
    slit1 = (x > 4.0) & (x < 5.0) & (x > 4.3) & (x < 4.55)
    slit2 = (x > 4.0) & (x < 5.0) & (x > 4.65) & (x < 4.9)
    V[wall] = 3.0
    V[slit1] = 0.0
    V[slit2] = 0.0
    return V

def potential_harmonic(x):
    return 0.02 * x ** 2

def potential_step(x):
    V = np.zeros_like(x)
    V[x > 2.0] = 0.4
    return V

POTENTIAL_FUNCS = [
    potential_free,
    potential_barrier,
    potential_double_slit,
    potential_harmonic,
    potential_step,
]

# ---------------------------------------------------------------------------
# Wavepacket initialization
# ---------------------------------------------------------------------------

def gaussian_wavepacket(x, x0=-8.0, k0=3.0, sigma=1.5):
    psi = np.exp(-((x - x0) ** 2) / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * DX)
    return psi

# ---------------------------------------------------------------------------
# Crank-Nicolson time evolution
# ---------------------------------------------------------------------------

def build_cn_matrices(V, dt, dx, nx):
    """Build tridiagonal matrices for Crank-Nicolson scheme."""
    r = 1j * dt / (4.0 * dx ** 2)
    diag_A = np.ones(nx, dtype=complex) * (1.0 + 2.0 * r + 1j * dt * V / 2.0)
    off_A = np.ones(nx - 1, dtype=complex) * (-r)
    diag_B = np.ones(nx, dtype=complex) * (1.0 - 2.0 * r - 1j * dt * V / 2.0)
    off_B = np.ones(nx - 1, dtype=complex) * r
    return diag_A, off_A, diag_B, off_B


def tridiag_solve(diag, off_lower, off_upper, rhs):
    """Thomas algorithm for tridiagonal system."""
    n = len(rhs)
    c = np.zeros(n, dtype=complex)
    d = np.zeros(n, dtype=complex)
    x = np.zeros(n, dtype=complex)

    c[0] = off_upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        ol = off_lower[i - 1] if i - 1 < len(off_lower) else 0
        m = diag[i] - ol * c[i - 1]
        c[i] = (off_upper[i] / m) if i < n - 1 else 0
        d[i] = (rhs[i] - ol * d[i - 1]) / m

    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def evolve_step(psi, diag_A, off_A, diag_B, off_B):
    """One Crank-Nicolson time step."""
    rhs = diag_B * psi
    rhs[:-1] += off_B * psi[1:]
    rhs[1:] += off_B * psi[:-1]
    return tridiag_solve(diag_A, off_A, off_A, rhs)


# ---------------------------------------------------------------------------
# UI Widgets
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, rect, text, color, active=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.active = active
        self.hovered = False

    def draw(self, surface, font):
        if self.active:
            col = self.color
            pygame.draw.rect(surface, col, self.rect, border_radius=6)
            txt_col = (0, 0, 0) if sum(col) > 400 else TITLE_COLOR
        else:
            col = (40, 45, 55) if not self.hovered else (55, 60, 72)
            pygame.draw.rect(surface, col, self.rect, border_radius=6)
            txt_col = TEXT_COLOR
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, 1, border_radius=6)
        txt = font.render(self.text, True, txt_col)
        surface.blit(txt, (self.rect.centerx - txt.get_width() // 2,
                           self.rect.centery - txt.get_height() // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return True
        return False


class Slider:
    def __init__(self, rect, min_val, max_val, value, label, color):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.color = color
        self.dragging = False

    @property
    def knob_x(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + int(ratio * self.rect.width)

    def draw(self, surface, font):
        lbl = font.render(f"{self.label}: {self.value:.2f}", True, self.color)
        surface.blit(lbl, (self.rect.x, self.rect.y - 20))
        ty = self.rect.centery
        pygame.draw.line(surface, BORDER_COLOR, (self.rect.x, ty),
                         (self.rect.right, ty), 3)
        pygame.draw.line(surface, self.color, (self.rect.x, ty),
                         (self.knob_x, ty), 3)
        pygame.draw.circle(surface, self.color, (self.knob_x, ty), 8)
        pygame.draw.circle(surface, TITLE_COLOR, (self.knob_x, ty), 8, 2)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if abs(event.pos[0] - self.knob_x) < 16 and abs(event.pos[1] - self.rect.centery) < 16:
                self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            ratio = (event.pos[0] - self.rect.x) / self.rect.width
            self.value = self.min_val + max(0, min(1, ratio)) * (self.max_val - self.min_val)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_glow(surface, pos, color, radius=14, intensity=60):
    gs = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
    for r in range(radius, 0, -2):
        a = int(intensity * (r / radius))
        pygame.draw.circle(gs, (*color, a), (radius * 2, radius * 2), r)
    surface.blit(gs, (pos[0] - radius * 2, pos[1] - radius * 2),
                 special_flags=pygame.BLEND_ADD)


def map_to_panel(x_val, y_val, panel, y_min, y_max):
    px = panel.x + int((x_val - X_MIN) / (X_MAX - X_MIN) * panel.width)
    py = panel.bottom - int((y_val - y_min) / (y_max - y_min) * panel.height)
    return max(panel.x, min(panel.right, px)), max(panel.y, min(panel.bottom, py))


def draw_wavefunction(surface, panel, psi, V, show_re, show_im, show_prob):
    re = np.real(psi)
    im = np.imag(psi)
    prob = np.abs(psi) ** 2

    all_vals = []
    if show_re:
        all_vals.extend([re.min(), re.max()])
    if show_im:
        all_vals.extend([im.min(), im.max()])
    if show_prob:
        all_vals.extend([0, prob.max()])
    all_vals.extend([0, V.max() * 1.1 if V.max() > 0 else 0.1])

    y_min = min(all_vals) - 0.05
    y_max = max(all_vals) + 0.05
    if y_max - y_min < 0.2:
        y_max = y_min + 0.2

    step = max(1, NX // panel.width)
    indices = range(0, NX, step)

    if V.max() > 0.001:
        v_points = []
        for i in indices:
            px, py = map_to_panel(X[i], V[i], panel, y_min, y_max)
            v_points.append((px, py))
        v_fill = list(v_points) + [(v_points[-1][0], panel.bottom), (v_points[0][0], panel.bottom)]
        if len(v_fill) > 2:
            vs = pygame.Surface((panel.width, panel.height), pygame.SRCALPHA)
            off = [(p[0] - panel.x, p[1] - panel.y) for p in v_fill]
            pygame.draw.polygon(vs, (250, 204, 21, 35), off)
            surface.blit(vs, panel.topleft)
        if len(v_points) > 1:
            pygame.draw.lines(surface, POTENTIAL_COLOR, False, v_points, 2)

    zero_y = map_to_panel(0, 0, panel, y_min, y_max)[1]
    pygame.draw.line(surface, (40, 45, 55), (panel.x, zero_y), (panel.right, zero_y), 1)

    if show_prob:
        pts = [map_to_panel(X[i], prob[i], panel, y_min, y_max) for i in indices]
        fill_pts = list(pts) + [(pts[-1][0], zero_y), (pts[0][0], zero_y)]
        if len(fill_pts) > 2:
            ps = pygame.Surface((panel.width, panel.height), pygame.SRCALPHA)
            off = [(p[0] - panel.x, p[1] - panel.y) for p in fill_pts]
            pygame.draw.polygon(ps, (*PROB_COLOR, 50), off)
            surface.blit(ps, panel.topleft)
        if len(pts) > 1:
            pygame.draw.lines(surface, PROB_COLOR, False, pts, 2)

    if show_re:
        pts = [map_to_panel(X[i], re[i], panel, y_min, y_max) for i in indices]
        if len(pts) > 1:
            pygame.draw.lines(surface, RE_COLOR, False, pts, 2)

    if show_im:
        pts = [map_to_panel(X[i], im[i], panel, y_min, y_max) for i in indices]
        if len(pts) > 1:
            pygame.draw.lines(surface, IM_COLOR, False, pts, 2)

    peak_idx = np.argmax(prob)
    peak_x, peak_y_val = X[peak_idx], prob[peak_idx]
    if peak_y_val > 0.005:
        ppx, ppy = map_to_panel(peak_x, peak_y_val, panel, y_min, y_max)
        draw_glow(surface, (ppx, ppy), PROB_COLOR, radius=12, intensity=50)


def draw_probability_bar(surface, rect, psi, V, font):
    """Draw a bar showing probability in left vs barrier vs right regions."""
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surface, BORDER_COLOR, rect, 1, border_radius=6)

    title = font.render("Probability Distribution", True, DIM_TEXT)
    surface.blit(title, (rect.x + 10, rect.y + 6))

    prob = np.abs(psi) ** 2 * DX

    barrier_mask = V > 0.01
    if barrier_mask.any():
        b_left = X[barrier_mask].min()
        b_right = X[barrier_mask].max()
    else:
        b_left = 0
        b_right = 0

    p_left = float(prob[X < b_left].sum()) if barrier_mask.any() else float(prob[X < 0].sum())
    p_barrier = float(prob[barrier_mask].sum()) if barrier_mask.any() else 0
    p_right = float(prob[X >= b_right].sum()) if barrier_mask.any() else float(prob[X >= 0].sum())

    bar_y = rect.y + 30
    bar_h = 20
    bar_x = rect.x + 15
    bar_w = rect.width - 30

    pygame.draw.rect(surface, (40, 45, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=4)

    w_left = int(p_left * bar_w)
    w_barrier = int(p_barrier * bar_w)
    w_right = bar_w - w_left - w_barrier

    if w_left > 0:
        pygame.draw.rect(surface, RE_COLOR, (bar_x, bar_y, w_left, bar_h), border_radius=4)
    if w_barrier > 0:
        pygame.draw.rect(surface, POTENTIAL_COLOR, (bar_x + w_left, bar_y, w_barrier, bar_h))
    if w_right > 0:
        pygame.draw.rect(surface, IM_COLOR, (bar_x + w_left + w_barrier, bar_y, w_right, bar_h),
                         border_radius=4)

    labels = [
        (RE_COLOR, f"Left: {p_left:.1%}"),
        (POTENTIAL_COLOR, f"Barrier: {p_barrier:.1%}"),
        (IM_COLOR, f"Right: {p_right:.1%}"),
    ]
    lx = bar_x
    for col, txt in labels:
        t = font.render(txt, True, col)
        surface.blit(t, (lx, bar_y + bar_h + 6))
        lx += t.get_width() + 20


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Time-Dependent Schrödinger Evolution")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("Menlo", 13)
        big_font = pygame.font.SysFont("Menlo", 18, bold=True)
        title_font = pygame.font.SysFont("Menlo", 24, bold=True)
    except Exception:
        font = pygame.font.Font(None, 17)
        big_font = pygame.font.Font(None, 22)
        title_font = pygame.font.Font(None, 28)

    wave_panel = pygame.Rect(20, 60, WIDTH - 40, 480)
    prob_rect = pygame.Rect(20, 560, 500, 80)
    legend_rect = pygame.Rect(540, 560, 260, 80)
    controls_y = 670

    current_scenario = 1
    dt = 0.03
    steps_per_frame = 4

    V = POTENTIAL_FUNCS[current_scenario](X)
    psi = gaussian_wavepacket(X)
    diag_A, off_A, diag_B, off_B = build_cn_matrices(V, dt, DX, NX)

    show_re = True
    show_im = True
    show_prob = True
    paused = False
    elapsed_time = 0.0

    scenario_btns = []
    bx = 20
    for i, name in enumerate(SCENARIOS):
        w = max(font.size(name)[0] + 20, 100)
        scenario_btns.append(Button((bx, controls_y, w, 32), name,
                                    PROB_COLOR, active=(i == current_scenario)))
        bx += w + 8

    toggle_re = Button((20, controls_y + 50, 80, 30), "Re[ψ]", RE_COLOR, active=show_re)
    toggle_im = Button((108, controls_y + 50, 80, 30), "Im[ψ]", IM_COLOR, active=show_im)
    toggle_prob = Button((196, controls_y + 50, 80, 30), "|ψ|²", PROB_COLOR, active=show_prob)
    pause_btn = Button((290, controls_y + 50, 80, 30), "Pause", (100, 110, 125))
    reset_btn = Button((378, controls_y + 50, 80, 30), "Reset", (100, 110, 125))

    k_slider = Slider((20, controls_y + 120, 300, 20), 0.5, 8.0, 3.0,
                       "k₀ (momentum)", RE_COLOR)
    sigma_slider = Slider((360, controls_y + 120, 300, 20), 0.5, 4.0, 1.5,
                           "σ (width)", IM_COLOR)
    speed_slider = Slider((700, controls_y + 120, 300, 20), 1.0, 15.0, 4.0,
                           "Speed", PROB_COLOR)

    def reset_sim():
        nonlocal psi, V, diag_A, off_A, diag_B, off_B, elapsed_time
        V = POTENTIAL_FUNCS[current_scenario](X)
        psi = gaussian_wavepacket(X, k0=k_slider.value, sigma=sigma_slider.value)
        diag_A, off_A, diag_B, off_B = build_cn_matrices(V, dt, DX, NX)
        elapsed_time = 0.0

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    pause_btn.text = "Play" if paused else "Pause"
                if event.key == pygame.K_r:
                    reset_sim()

            for i, btn in enumerate(scenario_btns):
                if btn.handle_event(event):
                    current_scenario = i
                    for j, b in enumerate(scenario_btns):
                        b.active = (j == i)
                    reset_sim()

            if toggle_re.handle_event(event):
                show_re = not show_re
                toggle_re.active = show_re
            if toggle_im.handle_event(event):
                show_im = not show_im
                toggle_im.active = show_im
            if toggle_prob.handle_event(event):
                show_prob = not show_prob
                toggle_prob.active = show_prob
            if pause_btn.handle_event(event):
                paused = not paused
                pause_btn.text = "Play" if paused else "Pause"
            if reset_btn.handle_event(event):
                reset_sim()

            k_slider.handle_event(event)
            sigma_slider.handle_event(event)
            speed_slider.handle_event(event)

        if not paused:
            steps = int(speed_slider.value)
            for _ in range(steps):
                psi = evolve_step(psi, diag_A, off_A, diag_B, off_B)
                elapsed_time += dt

        # --- Render ---
        screen.fill(BG_COLOR)

        t = title_font.render("Time-Dependent Schrödinger Evolution", True, TITLE_COLOR)
        screen.blit(t, (WIDTH // 2 - t.get_width() // 2, 16))
        sub = font.render(
            f"Scenario: {SCENARIOS[current_scenario]}  |  t = {elapsed_time:.2f}  |  "
            f"Norm = {np.sum(np.abs(psi)**2)*DX:.4f}",
            True, DIM_TEXT)
        screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 44))

        pygame.draw.rect(screen, PANEL_BG, wave_panel, border_radius=6)
        pygame.draw.rect(screen, BORDER_COLOR, wave_panel, 1, border_radius=6)

        inner = pygame.Rect(wave_panel.x + 5, wave_panel.y + 5,
                            wave_panel.width - 10, wave_panel.height - 10)
        draw_wavefunction(screen, inner, psi, V, show_re, show_im, show_prob)

        draw_probability_bar(screen, prob_rect, psi, V, font)

        pygame.draw.rect(screen, PANEL_BG, legend_rect, border_radius=6)
        pygame.draw.rect(screen, BORDER_COLOR, legend_rect, 1, border_radius=6)
        ly = legend_rect.y + 8
        for col, label in [(RE_COLOR, "Re[ψ(x,t)]"), (IM_COLOR, "Im[ψ(x,t)]"),
                           (PROB_COLOR, "|ψ(x,t)|²"), (POTENTIAL_COLOR, "V(x)")]:
            pygame.draw.line(screen, col, (legend_rect.x + 12, ly + 6),
                             (legend_rect.x + 30, ly + 6), 3)
            lt = font.render(label, True, col)
            screen.blit(lt, (legend_rect.x + 36, ly))
            ly += 17

        sc_label = font.render("Scenarios:", True, DIM_TEXT)
        screen.blit(sc_label, (20, controls_y - 18))
        for btn in scenario_btns:
            btn.draw(screen, font)

        toggle_re.draw(screen, font)
        toggle_im.draw(screen, font)
        toggle_prob.draw(screen, font)
        pause_btn.draw(screen, font)
        reset_btn.draw(screen, font)

        k_slider.draw(screen, font)
        sigma_slider.draw(screen, font)
        speed_slider.draw(screen, font)

        hints = font.render("[Space] Pause  [R] Reset  [Esc] Quit", True, DIM_TEXT)
        screen.blit(hints, (WIDTH - hints.get_width() - 20, HEIGHT - 22))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
