"""
Quantum Annealing vs. Gradient Descent Showdown
================================================
Real-time Pygame visualization comparing Simulated Quantum Annealing
with SGD on a multi-minima loss landscape. Demonstrates how quantum
tunneling escapes local minima that trap gradient descent.
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

SGD_COLOR = (255, 85, 85)
SGD_GLOW = (255, 85, 85)
QA_COLOR = (80, 250, 250)
QA_GLOW = (80, 250, 250)
LANDSCAPE_COLOR = (56, 132, 244)
LANDSCAPE_FILL = (30, 70, 140)
GLOBAL_MIN_COLOR = (255, 215, 0)

X_MIN, X_MAX = -8.0, 8.0
TRAIL_LENGTH = 80
TOTAL_STEPS = 400
PHYSICS_PER_FRAME = 1

# ---------------------------------------------------------------------------
# Loss landscape
# ---------------------------------------------------------------------------

def loss_landscape(x):
    baseline = 0.015 * x ** 2
    wells = (
        -2.0 * np.exp(-((x + 5.0) ** 2) / 1.0)
        - 4.5 * np.exp(-((x + 1.0) ** 2) / 0.5)
        - 2.0 * np.exp(-((x - 2.5) ** 2) / 0.6)
        - 8.0 * np.exp(-((x - 6.0) ** 2) / 1.2)
    )
    return baseline + wells


def loss_gradient(x, h=1e-5):
    return (loss_landscape(x + h) - loss_landscape(x - h)) / (2 * h)


xs_dense = np.linspace(X_MIN, X_MAX, 1000)
ys_dense = loss_landscape(xs_dense)
Y_MIN_VAL = float(ys_dense.min()) - 1.5
Y_MAX_VAL = float(ys_dense.max()) + 1.5

GLOBAL_MIN_X = float(xs_dense[np.argmin(ys_dense)])
GLOBAL_MIN_Y = float(ys_dense.min())

# Precompute barrier peaks (local maxima) for tunneling detection
grad_sign = np.sign(np.diff(ys_dense))
sign_changes = np.where(np.diff(grad_sign) < 0)[0]
BARRIER_XS = [float(xs_dense[i + 1]) for i in sign_changes]

# ---------------------------------------------------------------------------
# SGD Optimizer
# ---------------------------------------------------------------------------

class SGDOptimizer:
    def __init__(self, x0=-2.0, lr=0.012, momentum=0.9):
        self.x = x0
        self.v = 0.0
        self.lr = lr
        self.momentum = momentum
        self.history_x = [x0]
        self.history_loss = [float(loss_landscape(x0))]
        self.trapped = False
        self.trapped_counter = 0
        self.label = ""
        self.label_timer = 0

    def step(self):
        g = loss_gradient(self.x)
        self.v = self.momentum * self.v - self.lr * g
        self.x += self.v
        self.x = np.clip(self.x, X_MIN + 0.5, X_MAX - 0.5)
        loss = float(loss_landscape(self.x))
        self.history_x.append(self.x)
        self.history_loss.append(loss)

        if len(self.history_loss) > 30:
            recent = self.history_loss[-30:]
            if max(recent) - min(recent) < 0.05:
                self.trapped_counter += 1
            else:
                self.trapped_counter = 0

        if self.trapped_counter > 20 and not self.trapped:
            self.trapped = True
            self.label = "TRAPPED!"
            self.label_timer = 180

        if self.label_timer > 0:
            self.label_timer -= 1

    def reset(self, x0=-2.0):
        self.__init__(x0, self.lr, self.momentum)


# ---------------------------------------------------------------------------
# Quantum Annealer
# ---------------------------------------------------------------------------

class QuantumAnnealer:
    def __init__(self, x0=-2.0, gamma_start=5.0, gamma_end=0.01, total_steps=TOTAL_STEPS):
        self.x = x0
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.total_steps = total_steps
        self.step_count = 0
        self.gamma = gamma_start
        self.history_x = [x0]
        self.history_loss = [float(loss_landscape(x0))]
        self.tunneling = False
        self.tunnel_timer = 0
        self.tunnel_x = 0
        self.found_global = False
        self.label = ""
        self.label_timer = 0

    @property
    def wavefunction_width(self):
        return max(0.15, self.gamma * 0.6)

    def step(self):
        self.step_count += 1
        t = min(self.step_count / self.total_steps, 1.0)
        self.gamma = self.gamma_start * (self.gamma_end / self.gamma_start) ** t

        current_loss = loss_landscape(self.x)

        proposal_std = max(0.3, self.gamma * 1.2)
        x_new = self.x + np.random.normal(0, proposal_std)
        x_new = np.clip(x_new, X_MIN + 0.5, X_MAX - 0.5)
        new_loss = loss_landscape(x_new)

        delta = new_loss - current_loss
        if delta < 0:
            accept = True
        else:
            thermal_prob = math.exp(-delta / max(self.gamma, 0.01))

            barrier_factor = self._barrier_integral(self.x, x_new)
            tunnel_prob = math.exp(-barrier_factor / max(self.gamma, 0.01))

            accept = np.random.random() < max(thermal_prob, tunnel_prob)

        crossed_barrier = False
        if accept:
            x_old = self.x
            self.x = x_new
            for bx in BARRIER_XS:
                if (x_old < bx < x_new) or (x_new < bx < x_old):
                    crossed_barrier = True
                    break

        if crossed_barrier and delta > 0.5:
            self.tunneling = True
            self.tunnel_timer = 25
            self.tunnel_x = self.x
            self.label = "TUNNELING!"
            self.label_timer = 90

        if self.tunnel_timer > 0:
            self.tunnel_timer -= 1
            if self.tunnel_timer == 0:
                self.tunneling = False

        loss = float(loss_landscape(self.x))
        self.history_x.append(self.x)
        self.history_loss.append(loss)

        if abs(self.x - GLOBAL_MIN_X) < 0.8 and not self.found_global:
            self.found_global = True
            self.label = "GLOBAL MIN FOUND!"
            self.label_timer = 240

        if self.label_timer > 0:
            self.label_timer -= 1

    def _barrier_integral(self, x1, x2):
        n_samples = 20
        xs = np.linspace(x1, x2, n_samples)
        ys = loss_landscape(xs)
        current_e = loss_landscape(x1)
        barrier = np.maximum(ys - current_e, 0)
        integral = float(np.trapezoid(barrier, xs))
        return abs(integral)

    def reset(self, x0=-2.0):
        self.__init__(x0, self.gamma_start, self.gamma_end, self.total_steps)


# ---------------------------------------------------------------------------
# UI Widgets
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, rect, text, color, hover_color=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.hover_color = hover_color or tuple(min(c + 40, 255) for c in color)
        self.hovered = False

    def draw(self, surface, font):
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, 1, border_radius=6)
        txt = font.render(self.text, True, TITLE_COLOR)
        tx = self.rect.centerx - txt.get_width() // 2
        ty = self.rect.centery - txt.get_height() // 2
        surface.blit(txt, (tx, ty))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return True
        return False


class Slider:
    def __init__(self, rect, min_val, max_val, value, label):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.dragging = False

    @property
    def knob_x(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + int(ratio * self.rect.width)

    def draw(self, surface, font):
        lbl = font.render(f"{self.label}: {self.value:.1f}x", True, DIM_TEXT)
        surface.blit(lbl, (self.rect.x, self.rect.y - 20))
        track_y = self.rect.centery
        pygame.draw.line(surface, BORDER_COLOR, (self.rect.x, track_y),
                         (self.rect.right, track_y), 3)
        kx = self.knob_x
        pygame.draw.circle(surface, QA_COLOR, (kx, track_y), 8)
        pygame.draw.circle(surface, TITLE_COLOR, (kx, track_y), 8, 2)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            kx = self.knob_x
            ky = self.rect.centery
            if abs(event.pos[0] - kx) < 15 and abs(event.pos[1] - ky) < 15:
                self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            ratio = (event.pos[0] - self.rect.x) / self.rect.width
            ratio = max(0, min(1, ratio))
            self.value = self.min_val + ratio * (self.max_val - self.min_val)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def map_x_to_pixel(x, panel_rect):
    ratio = (x - X_MIN) / (X_MAX - X_MIN)
    return panel_rect.x + int(ratio * panel_rect.width)


def map_y_to_pixel(y, panel_rect):
    ratio = (y - Y_MIN_VAL) / (Y_MAX_VAL - Y_MIN_VAL)
    return panel_rect.bottom - int(ratio * panel_rect.height)


def draw_landscape(surface, panel_rect):
    points = []
    for i in range(len(xs_dense)):
        px = map_x_to_pixel(xs_dense[i], panel_rect)
        py = map_y_to_pixel(ys_dense[i], panel_rect)
        points.append((px, py))

    fill_points = list(points) + [
        (panel_rect.right, panel_rect.bottom),
        (panel_rect.x, panel_rect.bottom),
    ]
    if len(fill_points) > 2:
        fill_surf = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        offset_points = [(p[0] - panel_rect.x, p[1] - panel_rect.y) for p in fill_points]
        pygame.draw.polygon(fill_surf, (*LANDSCAPE_FILL, 100), offset_points)
        surface.blit(fill_surf, panel_rect.topleft)

    if len(points) > 1:
        pygame.draw.lines(surface, LANDSCAPE_COLOR, False, points, 2)

    gx = map_x_to_pixel(GLOBAL_MIN_X, panel_rect)
    gy = map_y_to_pixel(GLOBAL_MIN_Y, panel_rect)
    pygame.draw.circle(surface, GLOBAL_MIN_COLOR, (gx, gy), 5)


def draw_glow(surface, pos, color, radius=18, intensity=80):
    glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
    for r in range(radius, 0, -2):
        alpha = int(intensity * (r / radius))
        pygame.draw.circle(glow_surf, (*color, alpha),
                           (radius * 2, radius * 2), r)
    surface.blit(glow_surf,
                 (pos[0] - radius * 2, pos[1] - radius * 2),
                 special_flags=pygame.BLEND_ADD)


def draw_particle(surface, panel_rect, optimizer, color, glow_color):
    trail = optimizer.history_x[-TRAIL_LENGTH:]
    trail_losses = optimizer.history_loss[-TRAIL_LENGTH:]
    n = len(trail)

    for i in range(max(0, n - 1)):
        alpha = int(200 * ((i + 1) / n))
        px = map_x_to_pixel(trail[i], panel_rect)
        py = map_y_to_pixel(trail_losses[i], panel_rect)
        r = max(1, int(3 * ((i + 1) / n)))
        dot_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(dot_surf, (*color, alpha), (r + 1, r + 1), r)
        surface.blit(dot_surf, (px - r - 1, py - r - 1))

    cx = map_x_to_pixel(optimizer.history_x[-1], panel_rect)
    cy = map_y_to_pixel(optimizer.history_loss[-1], panel_rect)
    draw_glow(surface, (cx, cy), glow_color, radius=20, intensity=90)
    pygame.draw.circle(surface, color, (cx, cy), 6)
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 6, 2)


def draw_wavefunction(surface, panel_rect, annealer):
    cx = annealer.history_x[-1]
    cy = annealer.history_loss[-1]
    width = annealer.wavefunction_width

    if width < 0.2:
        return

    wave_surf = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    n_pts = 200
    xs = np.linspace(cx - 4 * width, cx + 4 * width, n_pts)
    gauss = np.exp(-((xs - cx) ** 2) / (2 * width ** 2))
    amplitude = 2.5 * (width / annealer.gamma_start) + 0.5

    points_top = []
    points_bot = []
    for i in range(n_pts):
        if xs[i] < X_MIN or xs[i] > X_MAX:
            continue
        px = map_x_to_pixel(xs[i], panel_rect) - panel_rect.x
        base_py = map_y_to_pixel(loss_landscape(xs[i]), panel_rect) - panel_rect.y
        offset = int(gauss[i] * amplitude * 40)
        points_top.append((px, base_py - offset))
        points_bot.append((px, base_py))

    if len(points_top) > 2:
        polygon = points_top + list(reversed(points_bot))
        alpha = min(120, int(60 + 60 * (width / annealer.gamma_start * 2)))
        pygame.draw.polygon(wave_surf, (*QA_COLOR, alpha), polygon)
        surface.blit(wave_surf, panel_rect.topleft)


def draw_tunnel_flash(surface, panel_rect, annealer):
    if annealer.tunnel_timer <= 0:
        return
    progress = annealer.tunnel_timer / 25.0
    cx = map_x_to_pixel(annealer.tunnel_x, panel_rect)
    cy = map_y_to_pixel(loss_landscape(annealer.tunnel_x), panel_rect)
    radius = int((1.0 - progress) * 60 + 10)
    alpha = int(progress * 180)

    flash_surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(flash_surf, (*QA_COLOR, alpha),
                       (radius + 2, radius + 2), radius, 3)
    inner_alpha = int(progress * 80)
    pygame.draw.circle(flash_surf, (*QA_COLOR, inner_alpha),
                       (radius + 2, radius + 2), radius)
    surface.blit(flash_surf, (cx - radius - 2, cy - radius - 2),
                 special_flags=pygame.BLEND_ADD)


def draw_loss_chart(surface, rect, sgd, qa, font):
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=4)
    pygame.draw.rect(surface, BORDER_COLOR, rect, 1, border_radius=4)

    title = font.render("Loss Over Time", True, DIM_TEXT)
    surface.blit(title, (rect.x + 10, rect.y + 5))

    padding = 30
    chart = pygame.Rect(rect.x + padding, rect.y + 25,
                        rect.width - padding * 2, rect.height - 35)

    max_steps = max(len(sgd.history_loss), len(qa.history_loss), 2)
    all_losses = sgd.history_loss + qa.history_loss
    if not all_losses:
        return
    loss_min = min(all_losses) - 0.5
    loss_max = max(all_losses) + 0.5
    if loss_max - loss_min < 1:
        loss_max = loss_min + 1

    def to_chart(step, loss):
        sx = chart.x + int((step / max(max_steps - 1, 1)) * chart.width)
        sy = chart.bottom - int(((loss - loss_min) / (loss_max - loss_min)) * chart.height)
        return (sx, max(chart.y, min(chart.bottom, sy)))

    if len(sgd.history_loss) > 1:
        pts = [to_chart(i, l) for i, l in enumerate(sgd.history_loss)]
        pygame.draw.lines(surface, SGD_COLOR, False, pts, 2)

    if len(qa.history_loss) > 1:
        pts = [to_chart(i, l) for i, l in enumerate(qa.history_loss)]
        pygame.draw.lines(surface, QA_COLOR, False, pts, 2)

    legend_y = rect.y + 5
    pygame.draw.line(surface, SGD_COLOR, (rect.right - 180, legend_y + 8),
                     (rect.right - 160, legend_y + 8), 2)
    sgd_lbl = font.render("SGD", True, SGD_COLOR)
    surface.blit(sgd_lbl, (rect.right - 155, legend_y))

    pygame.draw.line(surface, QA_COLOR, (rect.right - 100, legend_y + 8),
                     (rect.right - 80, legend_y + 8), 2)
    qa_lbl = font.render("QA", True, QA_COLOR)
    surface.blit(qa_lbl, (rect.right - 75, legend_y))


def draw_info_panel(surface, rect, sgd, qa, step, total, font, big_font):
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=4)
    pygame.draw.rect(surface, BORDER_COLOR, rect, 1, border_radius=4)

    y = rect.y + 10
    x = rect.x + 15

    step_txt = big_font.render(f"Step: {step} / {total}", True, TITLE_COLOR)
    surface.blit(step_txt, (x, y))
    y += 30

    gamma_txt = font.render(f"Gamma (QA): {qa.gamma:.3f}", True, QA_COLOR)
    surface.blit(gamma_txt, (x, y))
    y += 22

    bar_w = rect.width - 30
    bar_h = 10
    pygame.draw.rect(surface, BORDER_COLOR, (x, y, bar_w, bar_h), border_radius=3)
    fill_w = int(bar_w * (qa.gamma / qa.gamma_start))
    if fill_w > 0:
        pygame.draw.rect(surface, QA_COLOR, (x, y, fill_w, bar_h), border_radius=3)
    y += 22

    sgd_loss = sgd.history_loss[-1] if sgd.history_loss else 0
    qa_loss = qa.history_loss[-1] if qa.history_loss else 0
    sl = font.render(f"SGD Loss:  {sgd_loss:.3f}", True, SGD_COLOR)
    ql = font.render(f"QA  Loss:  {qa_loss:.3f}", True, QA_COLOR)
    surface.blit(sl, (x, y))
    y += 20
    surface.blit(ql, (x, y))
    y += 28

    sgd_pos = font.render(f"SGD  x = {sgd.history_x[-1]:.2f}", True, DIM_TEXT)
    qa_pos = font.render(f"QA   x = {qa.history_x[-1]:.2f}", True, DIM_TEXT)
    surface.blit(sgd_pos, (x, y))
    y += 18
    surface.blit(qa_pos, (x, y))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quantum Annealing vs. Gradient Descent Showdown")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("Menlo", 14)
        big_font = pygame.font.SysFont("Menlo", 18, bold=True)
        title_font = pygame.font.SysFont("Menlo", 26, bold=True)
        label_font = pygame.font.SysFont("Menlo", 22, bold=True)
    except Exception:
        font = pygame.font.Font(None, 18)
        big_font = pygame.font.Font(None, 22)
        title_font = pygame.font.Font(None, 30)
        label_font = pygame.font.Font(None, 26)

    left_panel = pygame.Rect(20, 70, 660, 420)
    right_panel = pygame.Rect(720, 70, 660, 420)
    chart_rect = pygame.Rect(20, 520, 800, 220)
    info_rect = pygame.Rect(840, 520, 310, 220)
    controls_rect = pygame.Rect(840, 760, 310, 120)

    sgd = SGDOptimizer(x0=-2.0)
    qa = QuantumAnnealer(x0=-2.0)

    play_btn = Button((controls_rect.x + 10, controls_rect.y + 10, 90, 36),
                      "Pause", (50, 60, 75))
    reset_btn = Button((controls_rect.x + 110, controls_rect.y + 10, 90, 36),
                       "Reset", (50, 60, 75))
    speed_slider = Slider((controls_rect.x + 10, controls_rect.y + 80,
                           controls_rect.width - 20, 20),
                          0.2, 5.0, 1.0, "Speed")

    running = True
    playing = True
    step = 0
    finished = False
    accumulator = 0.0

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    playing = not playing
                    play_btn.text = "Pause" if playing else "Play"
                if event.key == pygame.K_r:
                    sgd.reset()
                    qa.reset()
                    step = 0
                    finished = False

            if play_btn.handle_event(event):
                playing = not playing
                play_btn.text = "Pause" if playing else "Play"
            if reset_btn.handle_event(event):
                sgd.reset()
                qa.reset()
                step = 0
                finished = False
            speed_slider.handle_event(event)

        if playing and not finished:
            accumulator += speed_slider.value
            while accumulator >= 1.0 and step < TOTAL_STEPS:
                sgd.step()
                qa.step()
                step += 1
                accumulator -= 1.0
            if step >= TOTAL_STEPS:
                finished = True

        # --- Render ---
        screen.fill(BG_COLOR)

        title_surf = title_font.render(
            "Quantum Annealing  vs.  Gradient Descent", True, TITLE_COLOR)
        screen.blit(title_surf, (WIDTH // 2 - title_surf.get_width() // 2, 20))

        subtitle = font.render(
            "Same loss landscape  |  Same starting point  |  Different strategies",
            True, DIM_TEXT)
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 52))

        for panel, label_text, color in [
            (left_panel, "Stochastic Gradient Descent", SGD_COLOR),
            (right_panel, "Simulated Quantum Annealing", QA_COLOR),
        ]:
            pygame.draw.rect(screen, PANEL_BG, panel, border_radius=6)
            pygame.draw.rect(screen, BORDER_COLOR, panel, 1, border_radius=6)
            lbl = big_font.render(label_text, True, color)
            screen.blit(lbl, (panel.x + panel.width // 2 - lbl.get_width() // 2,
                              panel.y + 5))

        landscape_left = pygame.Rect(left_panel.x + 5, left_panel.y + 28,
                                     left_panel.width - 10, left_panel.height - 33)
        landscape_right = pygame.Rect(right_panel.x + 5, right_panel.y + 28,
                                      right_panel.width - 10, right_panel.height - 33)

        draw_landscape(screen, landscape_left)
        draw_landscape(screen, landscape_right)

        gmin_lbl = font.render("Global Min", True, GLOBAL_MIN_COLOR)
        gx_l = map_x_to_pixel(GLOBAL_MIN_X, landscape_left)
        gy_l = map_y_to_pixel(GLOBAL_MIN_Y, landscape_left)
        screen.blit(gmin_lbl, (gx_l - gmin_lbl.get_width() // 2, gy_l + 10))
        gx_r = map_x_to_pixel(GLOBAL_MIN_X, landscape_right)
        gy_r = map_y_to_pixel(GLOBAL_MIN_Y, landscape_right)
        screen.blit(gmin_lbl, (gx_r - gmin_lbl.get_width() // 2, gy_r + 10))

        draw_particle(screen, landscape_left, sgd, SGD_COLOR, SGD_GLOW)
        draw_wavefunction(screen, landscape_right, qa)
        draw_particle(screen, landscape_right, qa, QA_COLOR, QA_GLOW)
        draw_tunnel_flash(screen, landscape_right, qa)

        if sgd.label_timer > 0:
            alpha = min(255, sgd.label_timer * 4)
            lbl_surf = label_font.render(sgd.label, True, SGD_COLOR)
            lbl_alpha = pygame.Surface(lbl_surf.get_size(), pygame.SRCALPHA)
            lbl_alpha.fill((255, 255, 255, alpha))
            lbl_surf.blit(lbl_alpha, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            px = map_x_to_pixel(sgd.history_x[-1], landscape_left)
            py = map_y_to_pixel(sgd.history_loss[-1], landscape_left) - 35
            screen.blit(lbl_surf, (px - lbl_surf.get_width() // 2, py))

        if qa.label_timer > 0:
            alpha = min(255, qa.label_timer * 4)
            color = QA_COLOR if "TUNNEL" in qa.label else GLOBAL_MIN_COLOR
            lbl_surf = label_font.render(qa.label, True, color)
            lbl_alpha = pygame.Surface(lbl_surf.get_size(), pygame.SRCALPHA)
            lbl_alpha.fill((255, 255, 255, alpha))
            lbl_surf.blit(lbl_alpha, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            px = map_x_to_pixel(qa.history_x[-1], landscape_right)
            py = map_y_to_pixel(qa.history_loss[-1], landscape_right) - 35
            screen.blit(lbl_surf, (px - lbl_surf.get_width() // 2, py))

        draw_loss_chart(screen, chart_rect, sgd, qa, font)
        draw_info_panel(screen, info_rect, sgd, qa, step, TOTAL_STEPS, font, big_font)

        pygame.draw.rect(screen, PANEL_BG, controls_rect, border_radius=4)
        pygame.draw.rect(screen, BORDER_COLOR, controls_rect, 1, border_radius=4)
        ctrl_title = font.render("Controls  [Space] Play/Pause  [R] Reset",
                                 True, DIM_TEXT)
        screen.blit(ctrl_title, (controls_rect.x + 10, controls_rect.y - 16))
        play_btn.draw(screen, font)
        reset_btn.draw(screen, font)
        speed_slider.draw(screen, font)

        if finished:
            done_surf = big_font.render("Simulation Complete  —  Press R to restart",
                                        True, GLOBAL_MIN_COLOR)
            screen.blit(done_surf, (chart_rect.x + chart_rect.width // 2
                                    - done_surf.get_width() // 2,
                                    chart_rect.bottom + 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
