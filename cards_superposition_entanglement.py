"""
Superposition & Entanglement — Playing Card Analogy
====================================================
Interactive Pygame demo using a deck of cards to build intuition for
quantum superposition (one qubit) and entanglement (Bell pair).

No quantum math required on screen — cards carry the metaphor.
"""

import math
import random
import sys
import pygame

WIDTH, HEIGHT = 1200, 780
FPS = 60

BG = (15, 20, 28)
PANEL = (28, 34, 46)
BORDER = (55, 65, 82)
TEXT = (226, 232, 240)
DIM = (148, 163, 184)
TITLE = (248, 250, 252)
RED_SUIT = (239, 68, 68)
BLACK_SUIT = (148, 163, 184)
GOLD = (250, 204, 21)
CYAN = (34, 211, 238)
VIOLET = (167, 139, 250)

CARD_W, CARD_H = 140, 196


def draw_playing_card(surface, rect, suit: str, face_up: bool, pulse: float = 0.0):
    """suit: 'H' hearts (red) or 'S' spades (black). face_up False = card back."""
    x, y, w, h = rect.x, rect.y, rect.width, rect.height
    r = 12
    pygame.draw.rect(surface, (250, 250, 252), (x, y, w, h), border_radius=r)
    pygame.draw.rect(surface, BORDER, (x, y, w, h), 2, border_radius=r)
    if not face_up:
        # Card back pattern
        inner = pygame.Rect(x + 10, y + 10, w - 20, h - 20)
        pygame.draw.rect(surface, (30, 58, 95), inner, border_radius=8)
        for i in range(4):
            for j in range(6):
                cx = inner.x + 18 + i * 32
                cy = inner.y + 14 + j * 28
                s = 6 + int(3 * math.sin(pulse + i + j))
                pygame.draw.circle(surface, (59, 130, 246), (cx, cy), s, 1)
        return
    col = RED_SUIT if suit == "H" else BLACK_SUIT
    sym = "\u2665" if suit == "H" else "\u2660"
    try:
        font_big = pygame.font.SysFont("Menlo", 72, bold=True)
        font_small = pygame.font.SysFont("Menlo", 28, bold=True)
    except Exception:
        font_big = pygame.font.Font(None, 72)
        font_small = pygame.font.Font(None, 28)
    t = font_big.render(sym, True, col)
    surface.blit(t, (x + w // 2 - t.get_width() // 2, y + h // 2 - t.get_height() // 2))
    corner = font_small.render(sym, True, col)
    surface.blit(corner, (x + 10, y + 8))
    lbl = "|0⟩ ♥" if suit == "H" else "|1⟩ ♠"
    try:
        fl = pygame.font.SysFont("Menlo", 14)
    except Exception:
        fl = pygame.font.Font(None, 16)
    lt = fl.render(lbl, True, DIM)
    surface.blit(lt, (x + w // 2 - lt.get_width() // 2, y + h - 28))


def draw_superposed_card(surface, rect, theta: float, t_anim: float):
    """Blend two suits by Born weights for |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩."""
    p0 = math.cos(theta / 2) ** 2
    p1 = 1.0 - p0
    x, y, w, h = rect.x, rect.y, rect.width, rect.height
    a0 = int(70 + 110 * p0 * (0.5 + 0.5 * math.sin(t_anim * 3)))
    a1 = int(70 + 110 * p1 * (0.5 + 0.5 * math.sin(t_anim * 3 + 1.2)))
    blend = pygame.Surface((w, h), pygame.SRCALPHA)
    blend.fill((239, 68, 68, min(255, a0)))
    layer2 = pygame.Surface((w, h), pygame.SRCALPHA)
    layer2.fill((60, 70, 90, min(255, a1)))
    blend.blit(layer2, (0, 0))
    surface.blit(blend, (x, y))
    pygame.draw.rect(surface, GOLD, (x, y, w, h), 3, border_radius=12)
    try:
        fb = pygame.font.SysFont("Menlo", 56, bold=True)
        fs = pygame.font.SysFont("Menlo", 56, bold=True)
    except Exception:
        fb = fs = pygame.font.Font(None, 56)
    heart = fb.render("\u2665", True, RED_SUIT)
    spade = fs.render("\u2660", True, BLACK_SUIT)
    surface.blit(heart, (x + w // 4 - heart.get_width() // 2, y + h // 2 - heart.get_height() // 2))
    surface.blit(spade, (x + 3 * w // 4 - spade.get_width() // 2, y + h // 2 - spade.get_height() // 2))
    try:
        fl = pygame.font.SysFont("Menlo", 13)
    except Exception:
        fl = pygame.font.Font(None, 16)
    msg = fl.render("Superposition — not definitely ♥ or ♠ yet", True, GOLD)
    surface.blit(msg, (x + w // 2 - msg.get_width() // 2, y + h - 36))


class Button:
    def __init__(self, rect, text, color, font_size=16):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.hovered = False
        self.font_size = font_size

    def draw(self, surface, font):
        c = tuple(min(255, x + 35) for x in self.color) if self.hovered else self.color
        pygame.draw.rect(surface, c, self.rect, border_radius=8)
        pygame.draw.rect(surface, BORDER, self.rect, 1, border_radius=8)
        t = font.render(self.text, True, TITLE)
        surface.blit(t, (self.rect.centerx - t.get_width() // 2, self.rect.centery - t.get_height() // 2))

    def click(self, pos):
        return self.rect.collidepoint(pos)


class Slider:
    def __init__(self, rect, lo, hi, val, label):
        self.rect = pygame.Rect(rect)
        self.lo, self.hi = lo, hi
        self.val = val
        self.label = label
        self.drag = False

    def knob_x(self):
        t = (self.val - self.lo) / (self.hi - self.lo)
        return self.rect.x + int(t * self.rect.width)

    def draw(self, surface, font):
        surface.blit(font.render(self.label, True, CYAN), (self.rect.x, self.rect.y - 22))
        pygame.draw.line(surface, BORDER, (self.rect.x, self.rect.centery), (self.rect.right, self.rect.centery), 4)
        k = self.knob_x()
        pygame.draw.circle(surface, VIOLET, (k, self.rect.centery), 10)
        pygame.draw.circle(surface, TITLE, (k, self.rect.centery), 10, 2)

    def handle(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if abs(e.pos[0] - self.knob_x()) < 18 and abs(e.pos[1] - self.rect.centery) < 18:
                self.drag = True
        if e.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        if e.type == pygame.MOUSEMOTION and self.drag:
            t = (e.pos[0] - self.rect.x) / self.rect.width
            self.val = max(self.lo, min(self.hi, self.lo + t * (self.hi - self.lo)))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cards: Superposition & Entanglement")
    clock = pygame.time.Clock()
    try:
        font = pygame.font.SysFont("Menlo", 15)
        font_b = pygame.font.SysFont("Menlo", 20, bold=True)
        font_t = pygame.font.SysFont("Menlo", 26, bold=True)
    except Exception:
        font = pygame.font.Font(None, 18)
        font_b = pygame.font.Font(None, 22)
        font_t = pygame.font.Font(None, 30)

    mode = 0  # 0 superposition, 1 entanglement
    t_anim = 0.0

    # Superposition state
    theta = math.pi / 2
    sup_measured = False
    sup_outcome = "H"
    sup_counts = {"H": 0, "S": 0}

    # Entanglement: None=preparing, 'pair' ready face down, 'alice' alice shown, 'done' both
    ent_state = "idle"
    ent_pair = ("H", "H")  # both same
    ent_alice = "H"
    ent_bob_shown = False
    ent_stats = {"HH": 0, "SS": 0}
    ent_trials = 0

    tab_sup = Button((40, 52, 200, 40), "Superposition", (99, 102, 241))
    tab_ent = Button((250, 52, 200, 40), "Entanglement", (55, 65, 82))

    btn_measure_sup = Button((420, 620, 180, 44), "Measure (look)", (185, 80, 80))
    btn_reset_sup = Button((620, 620, 200, 44), "New superposition", (55, 120, 95))

    btn_prepare = Button((380, 580, 220, 44), "Shuffle Bell pair", (124, 58, 237))
    btn_measure_alice = Button((620, 580, 220, 44), "Measure Alice's card", (8, 145, 178))
    btn_reset_ent = Button((860, 580, 180, 44), "Reset stats", (75, 85, 100))

    slider = Slider((420, 540, 400, 20), 0.05, math.pi - 0.05, theta, "Mix angle θ (deck tilt)")

    running = True
    while running:
        clock.tick(FPS)
        t_anim += 0.016
        mx, my = pygame.mouse.get_pos()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            if e.type == pygame.MOUSEMOTION:
                tab_sup.hovered = tab_sup.rect.collidepoint(e.pos)
                tab_ent.hovered = tab_ent.rect.collidepoint(e.pos)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if tab_sup.click(e.pos):
                    mode = 0
                    tab_sup.color = (99, 102, 241)
                    tab_ent.color = (55, 65, 82)
                if tab_ent.click(e.pos):
                    mode = 1
                    tab_ent.color = (99, 102, 241)
                    tab_sup.color = (55, 65, 82)
                if mode == 0:
                    slider.handle(e)
                    if btn_measure_sup.click(e.pos) and not sup_measured:
                        p0 = math.cos(theta / 2) ** 2
                        sup_outcome = "H" if random.random() < p0 else "S"
                        sup_measured = True
                        sup_counts[sup_outcome] += 1
                    if btn_reset_sup.click(e.pos):
                        sup_measured = False
                if mode == 1:
                    if btn_prepare.click(e.pos):
                        ent_state = "ready"
                        ent_pair = ("H", "H") if random.random() < 0.5 else ("S", "S")
                        ent_bob_shown = False
                    if btn_measure_alice.click(e.pos) and ent_state == "ready":
                        ent_alice = ent_pair[0]
                        ent_state = "collapsed"
                        ent_bob_shown = True
                        key = "HH" if ent_pair[0] == "H" else "SS"
                        ent_stats[key] += 1
                        ent_trials += 1
                    if btn_reset_ent.click(e.pos):
                        ent_stats = {"HH": 0, "SS": 0}
                        ent_trials = 0
                        ent_state = "idle"

            if mode == 0 and e.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
                slider.handle(e)

        if mode == 0 and slider.drag:
            theta = slider.val

        screen.fill(BG)

        title = font_t.render("Deck of Cards: Superposition & Entanglement", True, TITLE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 12))

        tab_sup.draw(screen, font_b)
        tab_ent.draw(screen, font_b)

        deck_note = font.render(
            "Analogy: ♥ = quantum |0⟩   ♠ = quantum |1⟩   (red vs black like a half-deck label)",
            True,
            DIM,
        )
        screen.blit(deck_note, (WIDTH // 2 - deck_note.get_width() // 2, 100))

        if mode == 0:
            box = pygame.Rect(WIDTH // 2 - CARD_W // 2, 200, CARD_W, CARD_H)
            if sup_measured:
                draw_playing_card(screen, box, sup_outcome, True, t_anim)
            else:
                draw_superposed_card(screen, box, theta, t_anim)

            p0 = math.cos(theta / 2) ** 2
            p1 = 1 - p0
            lines = [
                "Classically, a face-down card is already ♥ or ♠ — you just don't know which.",
                "Quantum superposition: before measurement, the card is not secretly one or the other;",
                "  the deck is 'tilted' between outcomes — P(♥) and P(♠) follow the Born rule.",
                "",
                f"P(measure ♥) = cos²(θ/2) = {p0:.3f}     P(measure ♠) = sin²(θ/2) = {p1:.3f}",
                f"After many measurements: ♥ {sup_counts['H']} times   ♠ {sup_counts['S']} times",
            ]
            y = 430
            for line in lines:
                screen.blit(font.render(line, True, TEXT if line else DIM), (60, y))
                y += 22

            slider.draw(screen, font)
            btn_measure_sup.draw(screen, font_b)
            btn_reset_sup.draw(screen, font_b)

        else:
            # Entanglement
            left = pygame.Rect(280, 220, CARD_W, CARD_H)
            right = pygame.Rect(WIDTH - 280 - CARD_W, 220, CARD_W, CARD_H)

            screen.blit(font_b.render("Alice", True, CYAN), (left.centerx - 30, 190))
            screen.blit(font_b.render("Bob", True, CYAN), (right.centerx - 22, 190))

            if ent_state == "idle":
                draw_playing_card(screen, left, "H", False, t_anim)
                draw_playing_card(screen, right, "H", False, t_anim)
                idle = font_b.render('Click "Shuffle Bell pair" to deal an entangled pair', True, DIM)
                screen.blit(idle, (WIDTH // 2 - idle.get_width() // 2, 445))
            elif ent_state == "ready":
                draw_playing_card(screen, left, "H", False, t_anim)
                draw_playing_card(screen, right, "H", False, t_anim)
                mid = font_b.render("Entangled: both ♥ or both ♠ — undecided until someone looks", True, GOLD)
                screen.blit(mid, (WIDTH // 2 - mid.get_width() // 2, 445))
            else:
                draw_playing_card(screen, left, ent_alice, True, t_anim)
                draw_playing_card(screen, right, ent_pair[1], True, t_anim)
                flash = int(80 + 80 * abs(math.sin(t_anim * 8)))
                pygame.draw.rect(screen, (*GOLD, min(255, flash)), right, 4, border_radius=12)

            para = [
                "Two cards are prepared together in a Bell-like state: |♥♥⟩ + |♠♠⟩ (normalized).",
                "Neither card has a definite color until measured — like two sealed envelopes.",
                "When Alice measures her card, Bob's outcome is instantly correlated (both ♥ or both ♠).",
                "This correlation persists even if Bob is far away — that's entanglement (simplified deck story).",
                "",
                f"Trials: {ent_trials}   both ♥: {ent_stats['HH']}   both ♠: {ent_stats['SS']}  (always matched!)",
            ]
            y = 480
            for line in para:
                c = GOLD if "always matched" in line else TEXT
                screen.blit(font.render(line, True, c if isinstance(c, tuple) else TEXT), (60, y))
                y += 22

            btn_prepare.draw(screen, font_b)
            btn_measure_alice.draw(screen, font_b)
            btn_reset_ent.draw(screen, font_b)

        hint = font.render("[Esc] Quit  ·  Tab buttons switch modes", True, DIM)
        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT - 28))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
