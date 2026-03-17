"""
Kitaev Chain: Majorana Zero Modes
=================================
Interactive Pygame visualization of the 1D Kitaev chain model for
topological superconductivity. Demonstrates how Majorana zero modes
emerge at the chain edges in the topological phase — the principle
behind Microsoft's topological qubit.

Drag the sliders to cross the phase boundary and watch:
  - Majorana bonds reorganize from intra-site to inter-site pairing
  - Zero-energy modes appear in the energy spectrum
  - Edge-localized wavefunctions light up
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
N_SITES = 12

BG_COLOR = (13, 17, 23)
PANEL_BG = (22, 27, 34)
BORDER_COLOR = (48, 54, 61)
TEXT_COLOR = (201, 209, 217)
DIM_TEXT = (125, 133, 144)
TITLE_COLOR = (240, 246, 252)

MAJORANA_A_COLOR = (251, 146, 60)   # warm orange
MAJORANA_B_COLOR = (56, 189, 248)   # cool blue
BOND_INTRA_COLOR = (161, 98, 247)   # purple
BOND_INTER_COLOR = (6, 182, 212)    # cyan
GOLD = (250, 204, 21)
TOPO_COLOR = (6, 182, 212)
TRIVIAL_COLOR = (100, 110, 125)
SPECTRUM_COLOR = (129, 140, 248)
ZERO_MODE_COLOR = (250, 204, 21)
EDGE_GLOW_COLOR = (250, 204, 21)
WF_COLOR_TOPO = (6, 182, 212)
WF_COLOR_TRIV = (161, 98, 247)

# ---------------------------------------------------------------------------
# Physics: BdG Hamiltonian
# ---------------------------------------------------------------------------

def build_bdg_hamiltonian(n, mu, t, delta):
    """
    Build the 2N x 2N Bogoliubov-de Gennes Hamiltonian for the Kitaev chain.
    Basis ordering: (c_1, c_2, ..., c_N, c_1^dag, c_2^dag, ..., c_N^dag)
    """
    H = np.zeros((2 * n, 2 * n), dtype=np.float64)

    for i in range(n):
        H[i, i] = -mu / 2.0
        H[n + i, n + i] = mu / 2.0

    for i in range(n - 1):
        H[i, i + 1] = -t
        H[i + 1, i] = -t
        H[n + i, n + i + 1] = t
        H[n + i + 1, n + i] = t

        H[i, n + i + 1] = delta
        H[i + 1, n + i] = -delta
        H[n + i + 1, i] = delta
        H[n + i, i + 1] = -delta

    return H


def diagonalize(n, mu, t, delta):
    H = build_bdg_hamiltonian(n, mu, t, delta)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def is_topological(mu, t):
    return abs(mu) < 2.0 * abs(t)


def compute_site_weights(eigenvectors, n, mode_idx):
    """Compute |psi|^2 weight on each site for a given eigenmode."""
    vec = eigenvectors[:, mode_idx]
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = abs(vec[i]) ** 2 + abs(vec[n + i]) ** 2
    total = weights.sum()
    if total > 1e-12:
        weights /= total
    return weights


def majorana_bond_strengths(n, mu, t, delta):
    """
    Compute intra-site and inter-site Majorana coupling strengths
    for visualization. Returns (intra, inter) arrays.
    Intra-site: proportional to |mu|
    Inter-site: proportional to |t| and |delta|
    """
    mu_scale = abs(mu) / max(abs(mu) + 2 * abs(t), 1e-6)
    t_scale = (abs(t) + abs(delta)) / max(abs(mu) + abs(t) + abs(delta), 1e-6)

    intra = np.full(n, mu_scale)
    inter = np.full(n - 1, t_scale)
    return intra, inter


# ---------------------------------------------------------------------------
# Smooth interpolation state
# ---------------------------------------------------------------------------

class SmoothValue:
    def __init__(self, value, speed=0.08):
        self.target = value
        self.current = value
        self.speed = speed

    def set_target(self, v):
        self.target = v

    def update(self):
        self.current += (self.target - self.current) * self.speed

    @property
    def val(self):
        return self.current


# ---------------------------------------------------------------------------
# UI Widgets
# ---------------------------------------------------------------------------

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
        lbl = font.render(f"{self.label} = {self.value:.2f}", True, self.color)
        surface.blit(lbl, (self.rect.x, self.rect.y - 22))
        track_y = self.rect.centery
        pygame.draw.line(surface, BORDER_COLOR,
                         (self.rect.x, track_y), (self.rect.right, track_y), 3)
        filled_rect = pygame.Rect(self.rect.x, track_y - 1,
                                  self.knob_x - self.rect.x, 3)
        pygame.draw.rect(surface, self.color, filled_rect)
        kx = self.knob_x
        pygame.draw.circle(surface, self.color, (kx, track_y), 9)
        pygame.draw.circle(surface, TITLE_COLOR, (kx, track_y), 9, 2)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            kx = self.knob_x
            ky = self.rect.centery
            if abs(event.pos[0] - kx) < 16 and abs(event.pos[1] - ky) < 16:
                self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            ratio = (event.pos[0] - self.rect.x) / self.rect.width
            ratio = max(0.0, min(1.0, ratio))
            self.value = self.min_val + ratio * (self.max_val - self.min_val)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_glow(surface, pos, color, radius=16, intensity=70):
    glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
    for r in range(radius, 0, -2):
        alpha = int(intensity * (r / radius))
        pygame.draw.circle(glow_surf, (*color, alpha),
                           (radius * 2, radius * 2), r)
    surface.blit(glow_surf,
                 (pos[0] - radius * 2, pos[1] - radius * 2),
                 special_flags=pygame.BLEND_ADD)


def draw_chain(surface, rect, n, mu, t, delta, topo, pulse_phase,
               smooth_intra, smooth_inter, smooth_edge):
    """Draw the chain of Majorana operators with bonds."""
    pad_x = 60
    pad_y = 40
    usable_w = rect.width - 2 * pad_x
    spacing = usable_w / max(n - 1, 1)
    cy = rect.y + rect.height // 2
    majorana_sep = 14

    intra_strength, inter_strength = majorana_bond_strengths(n, mu, t, delta)

    positions_a = []
    positions_b = []
    for i in range(n):
        cx = rect.x + pad_x + int(i * spacing)
        positions_a.append((cx - majorana_sep, cy))
        positions_b.append((cx + majorana_sep, cy))

    for i in range(n):
        s = smooth_intra[i].val
        if s > 0.05:
            ax, ay = positions_a[i]
            bx, by = positions_b[i]
            alpha = int(s * 200)
            width = max(1, int(s * 4))
            line_surf = pygame.Surface((abs(bx - ax) + 4, 10), pygame.SRCALPHA)
            pygame.draw.line(line_surf, (*BOND_INTRA_COLOR, alpha),
                             (2, 5), (abs(bx - ax) + 2, 5), width)
            surface.blit(line_surf, (min(ax, bx) - 2, ay - 5))

    for i in range(n - 1):
        s = smooth_inter[i].val
        if s > 0.05:
            bx, by = positions_b[i]
            ax2, ay2 = positions_a[i + 1]
            pulse = 0.7 + 0.3 * math.sin(pulse_phase + i * 0.5)
            alpha = int(s * pulse * 220)
            width = max(1, int(s * 3.5))
            seg_surf = pygame.Surface(
                (abs(ax2 - bx) + 6, 12), pygame.SRCALPHA)
            pygame.draw.line(seg_surf, (*BOND_INTER_COLOR, alpha),
                             (3, 6), (abs(ax2 - bx) + 3, 6), width)
            surface.blit(seg_surf, (min(bx, ax2) - 3, by - 6))

    for i in range(n):
        ax, ay = positions_a[i]
        bx, by = positions_b[i]
        pygame.draw.circle(surface, MAJORANA_A_COLOR, (ax, ay), 8)
        pygame.draw.circle(surface, MAJORANA_B_COLOR, (bx, by), 8)
        pygame.draw.circle(surface, (255, 255, 255), (ax, ay), 8, 1)
        pygame.draw.circle(surface, (255, 255, 255), (bx, by), 8, 1)

    edge_alpha = smooth_edge.val
    if edge_alpha > 0.05:
        pulse_a = 0.6 + 0.4 * math.sin(pulse_phase * 1.3)
        glow_intensity = int(edge_alpha * pulse_a * 100)
        glow_r = int(20 + edge_alpha * 10)
        draw_glow(surface, positions_a[0], EDGE_GLOW_COLOR,
                  radius=glow_r, intensity=glow_intensity)
        draw_glow(surface, positions_b[-1], EDGE_GLOW_COLOR,
                  radius=glow_r, intensity=glow_intensity)
        ring_r = int(14 + 4 * math.sin(pulse_phase * 1.5))
        ring_alpha = int(edge_alpha * pulse_a * 200)
        ring_surf = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(ring_surf, (*EDGE_GLOW_COLOR, ring_alpha),
                           (ring_r + 2, ring_r + 2), ring_r, 2)
        for pos in [positions_a[0], positions_b[-1]]:
            surface.blit(ring_surf, (pos[0] - ring_r - 2, pos[1] - ring_r - 2))

    return positions_a, positions_b


def draw_energy_spectrum(surface, rect, eigenvalues, font):
    """Draw energy eigenvalue spectrum as horizontal lines."""
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surface, BORDER_COLOR, rect, 1, border_radius=6)

    title = font.render("Energy Spectrum", True, DIM_TEXT)
    surface.blit(title, (rect.x + 10, rect.y + 6))

    pad = 30
    plot_rect = pygame.Rect(rect.x + pad + 30, rect.y + 28,
                            rect.width - pad * 2 - 30, rect.height - 40)

    if len(eigenvalues) == 0:
        return

    e_max = max(abs(eigenvalues.max()), abs(eigenvalues.min()), 1.0)
    e_range = e_max * 1.1

    center_y = plot_rect.centery
    pygame.draw.line(surface, BORDER_COLOR,
                     (plot_rect.x, center_y), (plot_rect.right, center_y), 1)

    zero_lbl = font.render("E=0", True, DIM_TEXT)
    surface.blit(zero_lbl, (plot_rect.x - 28, center_y - 7))

    n_eig = len(eigenvalues)
    spacing = plot_rect.width / max(n_eig + 1, 1)

    for i, e in enumerate(eigenvalues):
        px = plot_rect.x + int((i + 1) * spacing)
        py = center_y - int((e / e_range) * (plot_rect.height // 2))
        py = max(plot_rect.y + 4, min(plot_rect.bottom - 4, py))

        is_zero = abs(e) < 0.15
        color = ZERO_MODE_COLOR if is_zero else SPECTRUM_COLOR
        width = 3 if is_zero else 2
        half_w = 6 if is_zero else 4

        pygame.draw.line(surface, color,
                         (px - half_w, py), (px + half_w, py), width)

        if is_zero:
            draw_glow(surface, (px, py), ZERO_MODE_COLOR,
                      radius=10, intensity=50)


def draw_wavefunction(surface, rect, weights, topo, font):
    """Draw |psi|^2 bar chart for the lowest-energy eigenstate."""
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surface, BORDER_COLOR, rect, 1, border_radius=6)

    title = font.render("Zero-Mode Wavefunction  |ψ|²", True, DIM_TEXT)
    surface.blit(title, (rect.x + 10, rect.y + 6))

    pad = 20
    plot_rect = pygame.Rect(rect.x + pad, rect.y + 30,
                            rect.width - pad * 2, rect.height - 44)

    if len(weights) == 0:
        return

    n = len(weights)
    w_max = max(weights.max(), 0.01)
    bar_w = max(4, (plot_rect.width - 10) // n - 4)
    total_w = n * (bar_w + 4) - 4
    start_x = plot_rect.x + (plot_rect.width - total_w) // 2

    color = WF_COLOR_TOPO if topo else WF_COLOR_TRIV

    for i in range(n):
        bx = start_x + i * (bar_w + 4)
        bar_h = int((weights[i] / w_max) * (plot_rect.height - 4))
        bar_h = max(2, bar_h)
        by = plot_rect.bottom - bar_h

        bar_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((*color, 180))
        surface.blit(bar_surf, (bx, by))

        if weights[i] / w_max > 0.5 and topo:
            draw_glow(surface, (bx + bar_w // 2, by),
                      EDGE_GLOW_COLOR, radius=10, intensity=40)

    pygame.draw.line(surface, BORDER_COLOR,
                     (plot_rect.x, plot_rect.bottom),
                     (plot_rect.right, plot_rect.bottom), 1)

    for i in range(n):
        bx = start_x + i * (bar_w + 4) + bar_w // 2
        lbl = font.render(str(i + 1), True, DIM_TEXT)
        surface.blit(lbl, (bx - lbl.get_width() // 2, plot_rect.bottom + 2))


def draw_phase_indicator(surface, rect, topo, smooth_topo, font, big_font):
    """Draw the TOPOLOGICAL / TRIVIAL phase banner."""
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surface, BORDER_COLOR, rect, 1, border_radius=6)

    t_val = smooth_topo.val
    r = int(TOPO_COLOR[0] * t_val + TRIVIAL_COLOR[0] * (1 - t_val))
    g = int(TOPO_COLOR[1] * t_val + TRIVIAL_COLOR[1] * (1 - t_val))
    b = int(TOPO_COLOR[2] * t_val + TRIVIAL_COLOR[2] * (1 - t_val))
    blended = (r, g, b)

    label = "TOPOLOGICAL PHASE" if topo else "TRIVIAL PHASE"
    txt = big_font.render(label, True, blended)
    surface.blit(txt, (rect.centerx - txt.get_width() // 2,
                       rect.centery - txt.get_height() // 2 - 10))

    sub = "Unpaired Majorana edge modes present" if topo else "All Majoranas paired on-site"
    sub_txt = font.render(sub, True, DIM_TEXT)
    surface.blit(sub_txt, (rect.centerx - sub_txt.get_width() // 2,
                           rect.centery + 10))

    if topo and t_val > 0.5:
        indicator_surf = pygame.Surface((rect.width - 4, rect.height - 4), pygame.SRCALPHA)
        alpha = int(t_val * 25)
        pygame.draw.rect(indicator_surf, (*TOPO_COLOR, alpha),
                         (0, 0, rect.width - 4, rect.height - 4),
                         border_radius=5)
        surface.blit(indicator_surf, (rect.x + 2, rect.y + 2))


def draw_legend(surface, rect, font):
    """Draw a small legend for Majorana colors."""
    y = rect.y + 8
    x = rect.x + 12

    items = [
        (MAJORANA_A_COLOR, "γ_A (Majorana A)"),
        (MAJORANA_B_COLOR, "γ_B (Majorana B)"),
        (BOND_INTRA_COLOR, "Intra-site bond"),
        (BOND_INTER_COLOR, "Inter-site bond"),
        (EDGE_GLOW_COLOR, "Zero mode (edge)"),
    ]
    for color, label in items:
        pygame.draw.circle(surface, color, (x + 5, y + 7), 5)
        txt = font.render(label, True, TEXT_COLOR)
        surface.blit(txt, (x + 16, y))
        y += 20


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Kitaev Chain — Majorana Zero Modes")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("Menlo", 13)
        big_font = pygame.font.SysFont("Menlo", 20, bold=True)
        title_font = pygame.font.SysFont("Menlo", 24, bold=True)
    except Exception:
        font = pygame.font.Font(None, 17)
        big_font = pygame.font.Font(None, 24)
        title_font = pygame.font.Font(None, 28)

    slider_mu = Slider((50, HEIGHT - 70, 320, 20), 0.0, 4.0, 0.5,
                        "μ (chemical potential)", MAJORANA_A_COLOR)
    slider_t = Slider((420, HEIGHT - 70, 320, 20), 0.0, 2.0, 1.0,
                       "t (hopping)", BOND_INTER_COLOR)
    slider_delta = Slider((790, HEIGHT - 70, 320, 20), 0.0, 2.0, 1.0,
                           "Δ (pairing gap)", SPECTRUM_COLOR)

    chain_rect = pygame.Rect(20, 60, WIDTH - 40, 320)
    spectrum_rect = pygame.Rect(20, 400, 500, 280)
    wf_rect = pygame.Rect(540, 400, 480, 280)
    phase_rect = pygame.Rect(1040, 400, 340, 140)
    legend_rect = pygame.Rect(1040, 555, 340, 125)

    smooth_intra = [SmoothValue(0.0, 0.07) for _ in range(N_SITES)]
    smooth_inter = [SmoothValue(0.0, 0.07) for _ in range(N_SITES - 1)]
    smooth_edge = SmoothValue(0.0, 0.06)
    smooth_topo = SmoothValue(0.0, 0.06)

    pulse_phase = 0.0
    running = True

    while running:
        clock.tick(FPS)
        pulse_phase += 0.05

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            slider_mu.handle_event(event)
            slider_t.handle_event(event)
            slider_delta.handle_event(event)

        mu = slider_mu.value
        t = slider_t.value
        delta = slider_delta.value
        topo = is_topological(mu, t) and delta > 0.01

        eigenvalues, eigenvectors = diagonalize(N_SITES, mu, t, delta)

        zero_indices = np.where(np.abs(eigenvalues) < 0.15)[0]
        if len(zero_indices) > 0:
            mid = len(eigenvalues) // 2
            mode_idx = mid
        else:
            mid = len(eigenvalues) // 2
            mode_idx = mid
        weights = compute_site_weights(eigenvectors, N_SITES, mode_idx)

        intra_s, inter_s = majorana_bond_strengths(N_SITES, mu, t, delta)
        for i in range(N_SITES):
            smooth_intra[i].set_target(intra_s[i])
            smooth_intra[i].update()
        for i in range(N_SITES - 1):
            smooth_inter[i].set_target(inter_s[i])
            smooth_inter[i].update()
        smooth_edge.set_target(1.0 if topo else 0.0)
        smooth_edge.update()
        smooth_topo.set_target(1.0 if topo else 0.0)
        smooth_topo.update()

        # --- Render ---
        screen.fill(BG_COLOR)

        title_surf = title_font.render(
            "Kitaev Chain  —  Majorana Zero Modes", True, TITLE_COLOR)
        screen.blit(title_surf, (WIDTH // 2 - title_surf.get_width() // 2, 18))

        sub = font.render(
            "Toy model for Microsoft's topological qubit  |  Drag sliders to explore phase transitions",
            True, DIM_TEXT)
        screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 46))

        pygame.draw.rect(screen, PANEL_BG, chain_rect, border_radius=6)
        pygame.draw.rect(screen, BORDER_COLOR, chain_rect, 1, border_radius=6)

        draw_chain(screen, chain_rect, N_SITES, mu, t, delta, topo,
                   pulse_phase, smooth_intra, smooth_inter, smooth_edge)

        chain_label = font.render(
            "Chain Sites (each split into γ_A and γ_B Majorana operators)",
            True, DIM_TEXT)
        screen.blit(chain_label,
                    (chain_rect.centerx - chain_label.get_width() // 2,
                     chain_rect.y + 8))

        for i in range(N_SITES):
            pad_x = 60
            spacing = (chain_rect.width - 2 * pad_x) / max(N_SITES - 1, 1)
            cx = chain_rect.x + pad_x + int(i * spacing)
            cy = chain_rect.y + chain_rect.height // 2
            lbl = font.render(str(i + 1), True, DIM_TEXT)
            screen.blit(lbl, (cx - lbl.get_width() // 2, cy + 22))

        draw_energy_spectrum(screen, spectrum_rect, eigenvalues, font)
        draw_wavefunction(screen, wf_rect, weights, topo, font)
        draw_phase_indicator(screen, phase_rect, topo, smooth_topo, font, big_font)

        pygame.draw.rect(screen, PANEL_BG, legend_rect, border_radius=6)
        pygame.draw.rect(screen, BORDER_COLOR, legend_rect, 1, border_radius=6)
        draw_legend(screen, legend_rect, font)

        topo_line_y = HEIGHT - 100
        boundary_label = font.render(
            f"|μ| = {abs(mu):.2f}   2t = {2*t:.2f}   →  "
            + ("TOPOLOGICAL (|μ| < 2t)" if topo else "TRIVIAL (|μ| ≥ 2t)"),
            True, TOPO_COLOR if topo else TRIVIAL_COLOR)
        screen.blit(boundary_label, (50, topo_line_y))

        slider_mu.draw(screen, font)
        slider_t.draw(screen, font)
        slider_delta.draw(screen, font)

        keys_hint = font.render("[Esc] Quit", True, DIM_TEXT)
        screen.blit(keys_hint, (WIDTH - keys_hint.get_width() - 20, HEIGHT - 24))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
