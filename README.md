# Aarnav Nagabhirava — Quantum Projects

Interactive quantum physics simulations built with Python and Pygame — from foundational concepts to advanced topics.

## Quick Start

```bash
pip install -r requirements.txt
python3 launcher.py
```

This opens a web dashboard at `http://localhost:8000` where you can launch any project with a single click.

---

## Foundations

### 1. Schrödinger Evolution

Numerically evolve a quantum wavepacket in real time using the Crank-Nicolson method. Visualize Re[ψ], Im[ψ], and |ψ|² simultaneously.

**Run directly:** `python3 schrodinger_evolution.py`

- Five potential scenarios: free propagation, barrier tunneling, double slit interference, harmonic oscillator, step potential
- Adjustable momentum (k₀), wavepacket width (σ), and simulation speed
- Probability distribution tracker showing left/barrier/right regions
- Toggle Re[ψ], Im[ψ], |ψ|² independently

### 2. Bloch Sphere & Qubit Gates

Visualize single-qubit states on an interactive 3D Bloch sphere with full mouse-drag rotation.

**Run directly:** `python3 bloch_sphere.py`

- Apply quantum gates (X, Y, Z, H, S, T) and watch animated rotations on the sphere
- Preset states: |0⟩, |1⟩, |+⟩, |−⟩, |+i⟩, |−i⟩
- Continuous Z-axis precession mode
- Live state vector readout with θ/φ angles and measurement probabilities
- Trail visualization showing the qubit's path on the sphere

---

## Advanced Topics

### 3. QA vs. Gradient Descent Showdown

Side-by-side comparison of Simulated Quantum Annealing vs. SGD on a multi-minima loss landscape. Watch quantum tunneling escape local minima that trap gradient descent.

**Run directly:** `python3 quantum_vs_gradient.py`

- Glowing particle trails with real-time loss chart
- Quantum wavefunction overlay and tunneling flash effects
- Interactive play/pause, reset, and speed controls

### 4. Kitaev Chain — Majorana Zero Modes

Interactive simulation of the 1D Kitaev chain model for topological superconductivity — the toy model behind Microsoft's topological qubit.

**Run directly:** `python3 kitaev_chain.py`

- 12-site chain with Majorana operator splitting (γ_A / γ_B)
- BdG Hamiltonian diagonalized in real time
- Energy spectrum with gold-highlighted zero modes
- Edge-localized wavefunction visualization
- Smooth phase transition animation between TOPOLOGICAL and TRIVIAL phases
