# Quantum Annealing vs. Gradient Descent Showdown

Real-time Pygame visualization comparing **Simulated Quantum Annealing** with **Stochastic Gradient Descent** on a multi-minima loss landscape. Watch quantum tunneling let the annealer escape local minima that permanently trap gradient descent.

## What You'll See

- **Left panel**: SGD with momentum descends the loss landscape and gets trapped in a local minimum
- **Right panel**: Quantum annealing explores via tunneling, escaping barriers to find the global minimum
- **Bottom**: Live loss-over-time chart tracking both methods, plus a Gamma decay gauge

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python quantum_vs_gradient.py
```

## Controls

| Key / Widget | Action |
|---|---|
| **Space** | Play / Pause |
| **R** | Reset simulation |
| **Esc** | Quit |
| **Speed slider** | Adjust simulation speed (0.2x - 5.0x) |
| **Play/Pause button** | Toggle simulation |
| **Reset button** | Restart from initial positions |

## How It Works

Both optimizers start at the same position (x = -6) on a crafted 1D loss landscape with 4 minima of varying depths.

**SGD** follows the gradient downhill with momentum. Once it settles into a local minimum, the gradient is near zero and it cannot escape — it's trapped.

**Quantum Annealing** maintains a "quantum fluctuation field" (Gamma) that starts large and decays over time. While Gamma is high, the annealer can propose large jumps and tunnel through energy barriers with nonzero probability (inspired by WKB tunneling). As Gamma decays, the annealer gradually collapses to the best minimum it has found — typically the global one.
