"""
Bloch Sphere: Qubit State Visualization
========================================
Interactive Pygame visualization of single-qubit states on the Bloch sphere.
Apply quantum gates (X, Y, Z, H, S, T) as rotations, watch precession,
and see measurement projections in real time.
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

SPHERE_COLOR = (60, 70, 90)
AXIS_COLOR = (80, 90, 110)
STATE_COLOR = (167, 139, 250)       # violet
STATE_GLOW = (167, 139, 250)
TRAIL_COLOR = (99, 179, 237)        # blue
X_AXIS_COLOR = (236, 121, 154)      # pink  (+x)
Y_AXIS_COLOR = (99, 179, 237)       # blue  (+y)
Z_AXIS_COLOR = (134, 239, 172)      # green (+z)
GATE_COLORS = {
    "X": (236, 121, 154),
    "Y": (99, 179, 237),
    "Z": (134, 239, 172),
    "H": (250, 204, 21),
    "S": (167, 139, 250),
    "T": (251, 146, 60),
}

SPHERE_RADIUS = 260
CAMERA_DIST = 6.0

# ---------------------------------------------------------------------------
# 3D projection
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self):
        self.theta = 0.35
        self.phi = -0.5
        self.center = (WIDTH // 2 - 100, HEIGHT // 2 - 20)

    def project(self, x, y, z):
        ct, st = math.cos(self.theta), math.sin(self.theta)
        cp, sp = math.cos(self.phi), math.sin(self.phi)

        xr = x * ct + z * st
        yr = x * st * sp + y * cp - z * ct * sp
        zr = -x * st * cp + y * sp + z * ct * cp

        scale = CAMERA_DIST / (CAMERA_DIST + zr + 1e-6)
        sx = int(self.center[0] + xr * SPHERE_RADIUS * scale)
        sy = int(self.center[1] - yr * SPHERE_RADIUS * scale)
        return sx, sy, scale

    def depth(self, x, y, z):
        ct, st = math.cos(self.theta), math.sin(self.theta)
        cp, sp = math.cos(self.phi), math.sin(self.phi)
        return -x * st * cp + y * sp + z * ct * cp


# ---------------------------------------------------------------------------
# Qubit state
# ---------------------------------------------------------------------------

class QubitState:
    def __init__(self):
        self.alpha = complex(1, 0)
        self.beta = complex(0, 0)
        self.trail = []
        self.max_trail = 300
        self.animating = False
        self.anim_gate = None
        self.anim_progress = 0.0
        self.anim_speed = 0.04
        self.anim_start_bloch = None
        self.anim_target_state = None

    @property
    def bloch(self):
        a, b = self.alpha, self.beta
        x = 2 * (a.conjugate() * b).real
        y = 2 * (a.conjugate() * b).imag
        z = abs(a) ** 2 - abs(b) ** 2
        return np.array([x, y, z])

    def set_state(self, alpha, beta):
        norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
        self.alpha = alpha / norm
        self.beta = beta / norm

    def apply_gate(self, gate_name):
        gates = {
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
            "H": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
            "S": np.array([[1, 0], [0, 1j]], dtype=complex),
            "T": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
        }
        if gate_name not in gates:
            return

        U = gates[gate_name]
        state = np.array([self.alpha, self.beta])
        new_state = U @ state

        self.anim_gate = gate_name
        self.anim_progress = 0.0
        self.anim_start_bloch = self.bloch.copy()
        self.anim_target_state = (new_state[0], new_state[1])
        self.animating = True

    def update(self):
        if self.animating:
            self.anim_progress += self.anim_speed
            if self.anim_progress >= 1.0:
                self.anim_progress = 1.0
                self.set_state(*self.anim_target_state)
                self.animating = False
                self.anim_gate = None
            else:
                target_alpha, target_beta = self.anim_target_state
                t = self.anim_progress
                t = t * t * (3 - 2 * t)  # smoothstep
                a = self.alpha * (1 - t) + target_alpha * t
                b = self.beta * (1 - t) + target_beta * t
                norm = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
                if norm > 1e-10:
                    a /= norm
                    b /= norm
                self.alpha, self.beta = a, b

        bv = self.bloch
        if len(self.trail) == 0 or np.linalg.norm(bv - self.trail[-1]) > 0.01:
            self.trail.append(bv.copy())
            if len(self.trail) > self.max_trail:
                self.trail.pop(0)

    def reset(self):
        self.alpha = complex(1, 0)
        self.beta = complex(0, 0)
        self.trail.clear()
        self.animating = False

    def set_preset(self, name):
        presets = {
            "|0⟩": (1, 0),
            "|1⟩": (0, 1),
            "|+⟩": (1 / math.sqrt(2), 1 / math.sqrt(2)),
            "|-⟩": (1 / math.sqrt(2), -1 / math.sqrt(2)),
            "|+i⟩": (1 / math.sqrt(2), 1j / math.sqrt(2)),
            "|-i⟩": (1 / math.sqrt(2), -1j / math.sqrt(2)),
        }
        if name in presets:
            a, b = presets[name]
            self.set_state(complex(a), complex(b))
            self.trail.clear()


# ---------------------------------------------------------------------------
# UI
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


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_sphere_wireframe(surface, cam):
    n_lines = 8
    pts_per_line = 60

    for i in range(n_lines):
        angle = i * math.pi / n_lines
        points = []
        for j in range(pts_per_line + 1):
            t = j / pts_per_line * 2 * math.pi
            x = math.cos(t) * math.cos(angle)
            y = math.cos(t) * math.sin(angle)
            z = math.sin(t)
            sx, sy, sc = cam.project(x, y, z)
            d = cam.depth(x, y, z)
            points.append((sx, sy, d))
        for j in range(len(points) - 1):
            a, b = points[j], points[j + 1]
            alpha = 0.15 if a[2] < 0 else 0.35
            color = tuple(int(c * alpha) for c in (180, 190, 210))
            pygame.draw.line(surface, color, (a[0], a[1]), (b[0], b[1]), 1)

    for zi in [-0.5, 0, 0.5]:
        r = math.sqrt(1 - zi ** 2)
        points = []
        for j in range(pts_per_line + 1):
            t = j / pts_per_line * 2 * math.pi
            x = r * math.cos(t)
            y = r * math.sin(t)
            sx, sy, sc = cam.project(x, y, zi)
            d = cam.depth(x, y, zi)
            points.append((sx, sy, d))
        for j in range(len(points) - 1):
            a, b = points[j], points[j + 1]
            alpha = 0.12 if a[2] < 0 else 0.3
            color = tuple(int(c * alpha) for c in (180, 190, 210))
            pygame.draw.line(surface, color, (a[0], a[1]), (b[0], b[1]), 1)


def draw_axes(surface, cam, font):
    axis_len = 1.25
    axes = [
        ((axis_len, 0, 0), X_AXIS_COLOR, "+X"),
        ((0, axis_len, 0), Y_AXIS_COLOR, "+Y"),
        ((0, 0, axis_len), Z_AXIS_COLOR, "|0⟩"),
        ((0, 0, -axis_len), Z_AXIS_COLOR, "|1⟩"),
    ]
    for (ax, ay, az), color, label in axes:
        sx, sy, sc = cam.project(ax, ay, az)
        ox, oy, _ = cam.project(0, 0, 0)
        d = cam.depth(ax, ay, az)
        alpha = 0.4 if d < 0 else 0.9
        c = tuple(int(v * alpha) for v in color)
        pygame.draw.line(surface, c, (ox, oy), (sx, sy), 2)
        pygame.draw.circle(surface, c, (sx, sy), 4)
        lbl = font.render(label, True, c)
        surface.blit(lbl, (sx + 8, sy - 8))


def draw_state_vector(surface, cam, qubit):
    bv = qubit.bloch
    sx, sy, sc = cam.project(bv[0], bv[1], bv[2])
    ox, oy, _ = cam.project(0, 0, 0)

    pygame.draw.line(surface, STATE_COLOR, (ox, oy), (sx, sy), 3)

    glow_surf = pygame.Surface((50, 50), pygame.SRCALPHA)
    for r in range(18, 0, -2):
        a = int(70 * (r / 18))
        pygame.draw.circle(glow_surf, (*STATE_GLOW, a), (25, 25), r)
    surface.blit(glow_surf, (sx - 25, sy - 25), special_flags=pygame.BLEND_ADD)
    pygame.draw.circle(surface, STATE_COLOR, (sx, sy), 7)
    pygame.draw.circle(surface, (255, 255, 255), (sx, sy), 7, 2)


def draw_trail(surface, cam, trail):
    n = len(trail)
    for i in range(max(0, n - 1)):
        a = trail[i]
        b = trail[i + 1] if i + 1 < n else trail[i]
        sx1, sy1, _ = cam.project(a[0], a[1], a[2])
        sx2, sy2, _ = cam.project(b[0], b[1], b[2])
        alpha = int(180 * ((i + 1) / n))
        ts = pygame.Surface((abs(sx2 - sx1) + 6, abs(sy2 - sy1) + 6), pygame.SRCALPHA)
        ox = min(sx1, sx2) - 3
        oy = min(sy1, sy2) - 3
        pygame.draw.line(ts, (*TRAIL_COLOR, alpha),
                         (sx1 - ox, sy1 - oy), (sx2 - ox, sy2 - oy), 2)
        surface.blit(ts, (ox, oy))


def draw_state_info(surface, rect, qubit, font, big_font):
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surface, BORDER_COLOR, rect, 1, border_radius=6)

    x, y_coord = rect.x + 15, rect.y + 12
    title = big_font.render("Qubit State", True, TITLE_COLOR)
    surface.blit(title, (x, y_coord))
    y_coord += 28

    a, b = qubit.alpha, qubit.beta
    state_str = f"|ψ⟩ = ({a.real:+.3f}{a.imag:+.3f}i)|0⟩ + ({b.real:+.3f}{b.imag:+.3f}i)|1⟩"
    st = font.render(state_str, True, STATE_COLOR)
    surface.blit(st, (x, y_coord))
    y_coord += 22

    bv = qubit.bloch
    bloch_str = f"Bloch: ({bv[0]:+.3f}, {bv[1]:+.3f}, {bv[2]:+.3f})"
    bt = font.render(bloch_str, True, DIM_TEXT)
    surface.blit(bt, (x, y_coord))
    y_coord += 22

    theta = math.acos(max(-1, min(1, bv[2])))
    phi = math.atan2(bv[1], bv[0])
    angles = f"θ = {math.degrees(theta):.1f}°   φ = {math.degrees(phi):.1f}°"
    at = font.render(angles, True, DIM_TEXT)
    surface.blit(at, (x, y_coord))
    y_coord += 22

    p0 = abs(a) ** 2
    p1 = abs(b) ** 2
    prob_str = f"P(|0⟩) = {p0:.3f}   P(|1⟩) = {p1:.3f}"
    pt = font.render(prob_str, True, TEXT_COLOR)
    surface.blit(pt, (x, y_coord))
    y_coord += 24

    bar_w = rect.width - 30
    bar_h = 12
    pygame.draw.rect(surface, (40, 45, 55), (x, y_coord, bar_w, bar_h), border_radius=4)
    w0 = int(p0 * bar_w)
    if w0 > 0:
        pygame.draw.rect(surface, Z_AXIS_COLOR, (x, y_coord, w0, bar_h), border_radius=4)
    if bar_w - w0 > 0:
        pygame.draw.rect(surface, X_AXIS_COLOR, (x + w0, y_coord, bar_w - w0, bar_h),
                         border_radius=4)
    y_coord += bar_h + 4
    l0 = font.render("|0⟩", True, Z_AXIS_COLOR)
    l1 = font.render("|1⟩", True, X_AXIS_COLOR)
    surface.blit(l0, (x, y_coord))
    surface.blit(l1, (x + bar_w - l1.get_width(), y_coord))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bloch Sphere — Qubit State Visualization")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("Menlo", 13)
        big_font = pygame.font.SysFont("Menlo", 18, bold=True)
        title_font = pygame.font.SysFont("Menlo", 24, bold=True)
    except Exception:
        font = pygame.font.Font(None, 17)
        big_font = pygame.font.Font(None, 22)
        title_font = pygame.font.Font(None, 28)

    cam = Camera()
    qubit = QubitState()

    info_rect = pygame.Rect(WIDTH - 380, 70, 360, 210)

    gate_names = ["X", "Y", "Z", "H", "S", "T"]
    gate_btns = []
    gx = WIDTH - 380
    gy = 300
    for i, name in enumerate(gate_names):
        gate_btns.append(Button((gx + i * 58, gy, 50, 36), name,
                                GATE_COLORS[name]))
        gate_btns[-1].handle_event(pygame.event.Event(0))

    preset_names = ["|0⟩", "|1⟩", "|+⟩", "|-⟩", "|+i⟩", "|-i⟩"]
    preset_btns = []
    px_start = WIDTH - 380
    py = 370
    for i, name in enumerate(preset_names):
        w = max(font.size(name)[0] + 16, 48)
        preset_btns.append(Button((px_start, py, w, 30), name, (80, 90, 110)))
        px_start += w + 6

    reset_btn = Button((WIDTH - 380, 420, 80, 32), "Reset", (100, 110, 125))
    clear_trail_btn = Button((WIDTH - 290, 420, 100, 32), "Clear Trail", (100, 110, 125))

    precessing = False
    precess_btn = Button((WIDTH - 180, 420, 100, 32), "Precess", (134, 239, 172))
    precess_axis = np.array([0.0, 0.0, 1.0])
    precess_speed = 0.02

    dragging = False
    last_mouse = (0, 0)

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    qubit.reset()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                in_ui = (info_rect.collidepoint(event.pos) or
                         any(b.rect.collidepoint(event.pos) for b in gate_btns) or
                         any(b.rect.collidepoint(event.pos) for b in preset_btns) or
                         reset_btn.rect.collidepoint(event.pos) or
                         clear_trail_btn.rect.collidepoint(event.pos) or
                         precess_btn.rect.collidepoint(event.pos))
                if not in_ui:
                    dragging = True
                    last_mouse = event.pos

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

            if event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                cam.theta += dx * 0.005
                cam.phi -= dy * 0.005
                cam.phi = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, cam.phi))
                last_mouse = event.pos

            for i, btn in enumerate(gate_btns):
                if btn.handle_event(event) and not qubit.animating:
                    qubit.apply_gate(gate_names[i])

            for i, btn in enumerate(preset_btns):
                if btn.handle_event(event):
                    qubit.set_preset(preset_names[i])

            if reset_btn.handle_event(event):
                qubit.reset()
            if clear_trail_btn.handle_event(event):
                qubit.trail.clear()
            if precess_btn.handle_event(event):
                precessing = not precessing
                precess_btn.active = precessing
                precess_btn.text = "Stop" if precessing else "Precess"

        qubit.update()

        if precessing and not qubit.animating:
            bv = qubit.bloch
            axis = precess_axis
            c = math.cos(precess_speed)
            s = math.sin(precess_speed)
            nx, ny, nz = axis
            R = np.array([
                [c + nx*nx*(1-c),     nx*ny*(1-c) - nz*s, nx*nz*(1-c) + ny*s],
                [ny*nx*(1-c) + nz*s,  c + ny*ny*(1-c),     ny*nz*(1-c) - nx*s],
                [nz*nx*(1-c) - ny*s,  nz*ny*(1-c) + nx*s,  c + nz*nz*(1-c)],
            ])
            new_bv = R @ bv
            theta = math.acos(max(-1, min(1, new_bv[2])))
            phi = math.atan2(new_bv[1], new_bv[0])
            qubit.set_state(math.cos(theta / 2),
                            math.sin(theta / 2) * np.exp(1j * phi))

        # --- Render ---
        screen.fill(BG_COLOR)

        t = title_font.render("Bloch Sphere  —  Qubit State Visualization", True, TITLE_COLOR)
        screen.blit(t, (WIDTH // 2 - t.get_width() // 2, 16))
        sub = font.render(
            "Drag to rotate  |  Click gates to apply  |  Watch the state vector move",
            True, DIM_TEXT)
        screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 44))

        draw_sphere_wireframe(screen, cam)
        draw_axes(screen, cam, font)
        draw_trail(screen, cam, qubit.trail)
        draw_state_vector(screen, cam, qubit)

        draw_state_info(screen, info_rect, qubit, font, big_font)

        gate_label = big_font.render("Quantum Gates:", True, DIM_TEXT)
        screen.blit(gate_label, (WIDTH - 380, gy - 22))
        for btn in gate_btns:
            btn.draw(screen, font)

        preset_label = font.render("Preset States:", True, DIM_TEXT)
        screen.blit(preset_label, (WIDTH - 380, py - 18))
        for btn in preset_btns:
            btn.draw(screen, font)

        reset_btn.draw(screen, font)
        clear_trail_btn.draw(screen, font)
        precess_btn.draw(screen, font)

        if qubit.animating and qubit.anim_gate:
            anim_txt = big_font.render(
                f"Applying {qubit.anim_gate} gate...", True,
                GATE_COLORS.get(qubit.anim_gate, STATE_COLOR))
            screen.blit(anim_txt, (WIDTH - 380, 460))

        explain_y = 500
        explain_rect = pygame.Rect(WIDTH - 380, explain_y, 360, 280)
        pygame.draw.rect(screen, PANEL_BG, explain_rect, border_radius=6)
        pygame.draw.rect(screen, BORDER_COLOR, explain_rect, 1, border_radius=6)
        ey = explain_y + 10
        ex = WIDTH - 368
        et = big_font.render("Gate Reference", True, TITLE_COLOR)
        screen.blit(et, (ex, ey)); ey += 26
        gate_info = [
            ("X", "Bit-flip: |0⟩↔|1⟩ (π rot around X)"),
            ("Y", "Bit+phase flip (π rot around Y)"),
            ("Z", "Phase-flip: |1⟩→-|1⟩ (π rot around Z)"),
            ("H", "Hadamard: creates superposition"),
            ("S", "Phase gate: |1⟩→i|1⟩ (π/2 around Z)"),
            ("T", "T gate: |1⟩→e^{iπ/4}|1⟩ (π/4 around Z)"),
        ]
        for name, desc in gate_info:
            col = GATE_COLORS[name]
            nt = font.render(f"{name}:", True, col)
            dt_txt = font.render(desc, True, DIM_TEXT)
            screen.blit(nt, (ex, ey))
            screen.blit(dt_txt, (ex + 25, ey))
            ey += 18

        ey += 10
        hint_lines = [
            "Drag mouse to rotate the sphere view",
            "Click gates to see qubit rotations",
            "Precess: continuous Z-axis rotation",
        ]
        for line in hint_lines:
            ht = font.render(line, True, DIM_TEXT)
            screen.blit(ht, (ex, ey)); ey += 16

        hints = font.render("[R] Reset  [Esc] Quit", True, DIM_TEXT)
        screen.blit(hints, (WIDTH - hints.get_width() - 20, HEIGHT - 22))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
