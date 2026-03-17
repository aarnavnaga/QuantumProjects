# Aarnav Nagabhirava — Quantum Projects

Interactive physics simulations built with Python and Pygame exploring quantum computing concepts.

## Quick Start

```bash
pip install -r requirements.txt
python3 launcher.py
```

This opens a web dashboard at `http://localhost:8000` where you can launch either project with a single click.

---

## Project 1: QA vs. Gradient Descent Showdown

Real-time visualization comparing **Simulated Quantum Annealing** with **Stochastic Gradient Descent** on a multi-minima loss landscape. Watch quantum tunneling let the annealer escape local minima that permanently trap gradient descent.

**Run directly:** `python3 quantum_vs_gradient.py`

### What You'll See

- **Left panel**: SGD with momentum descends the loss landscape and gets trapped in a local minimum
- **Right panel**: Quantum annealing explores via tunneling, escaping barriers to find the global minimum
- **Bottom**: Live loss-over-time chart tracking both methods, plus a Gamma decay gauge

### Controls

| Key / Widget | Action |
|---|---|
| **Space** | Play / Pause |
| **R** | Reset simulation |
| **Esc** | Quit |
| **Speed slider** | Adjust simulation speed (0.2x - 5.0x) |

### How It Works

Both optimizers start at the same position on a crafted 1D loss landscape with 4 minima of varying depths. SGD follows the gradient downhill with momentum and gets trapped. Quantum Annealing maintains a "quantum fluctuation field" (Gamma) that enables tunneling through energy barriers, gradually collapsing to the global minimum.

---

## Project 2: Kitaev Chain — Majorana Zero Modes

Interactive simulation of the 1D Kitaev chain model for topological superconductivity — the toy model behind Microsoft's topological qubit. Drag sliders across the phase boundary and watch Majorana zero modes emerge at the chain edges.

**Run directly:** `python3 kitaev_chain.py`

### What You'll See

- **Chain panel**: 12 sites, each split into two Majorana operators (γ_A orange, γ_B blue) with animated bonds
- **Energy spectrum**: Eigenvalues of the BdG Hamiltonian — zero-energy modes highlighted in gold in the topological phase
- **Wavefunction plot**: |ψ|² of the lowest-energy eigenstate showing edge localization
- **Phase indicator**: Real-time TOPOLOGICAL / TRIVIAL classification

### Controls

| Widget | Action |
|---|---|
| **μ slider** | Chemical potential (0 – 4) |
| **t slider** | Hopping energy (0 – 2) |
| **Δ slider** | Pairing gap (0 – 2) |
| **Esc** | Quit |

### How It Works

The Kitaev chain Hamiltonian is built as a Bogoliubov-de Gennes matrix and diagonalized in real time. When |μ| < 2t and Δ ≠ 0, the system enters the **topological phase**: Majorana operators pair between neighboring sites instead of within them, leaving two unpaired zero-energy modes localized at the chain edges. These are the Majorana zero modes — non-abelian anyons that could form the basis of fault-tolerant quantum computing.
