# Senior Thesis

A high-performance, pygame-based boids flocking simulation capable of
running **thousands of agents** in real time on a single core.

## Overview

This project simulates emergent flocking behavior using [Craig Reynolds'
boids model](https://en.wikipedia.org/wiki/Boids). Each boid independently
combines three local steering rules to produce realistic group motion:

- **Separation** – steer away from nearby boids to avoid crowding
- **Alignment** – match the average heading of local neighbors
- **Cohesion** – steer toward the center of mass of the local flock

The simulation runs **fullscreen** and colors each boid by its travel
direction, making the emergent alignment of flocks visually obvious.

## Requirements

- Python 3.10+
- [pygame](https://www.pygame.org/)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install pygame
```

## Running

```bash
python main.py
```

## Controls

| Key     | Action                    |
| ------- | ------------------------- |
| `SPACE` | Reset the flock           |
| `UP`    | Add 100 boids             |
| `DOWN`  | Remove 100 boids          |
| `ESC`   | Quit the simulation       |

Use `UP` and `DOWN` to stress-test the simulation; pushing toward
thousands of boids is the intended usage.

## Architecture

The monolithic, O(N²) brute-force implementation is replaced with
several complementary optimizations:

### Spatial Hash Grid

A uniform grid divides the world into cells of `PERCEPTION_RADIUS` width.
Each frame, the grid is rebuilt in O(N), and each boid queries only the
**3×3 cell neighborhood** around itself instead of scanning the entire
flock. This reduces neighbor search from O(N²) to ~O(N · k), where k is
the (small, roughly constant) average number of neighbors. Toroidal
wrapping is handled by modular cell indexing and shortest-vector
distance arithmetic.

### Single-Pass Neighbor Analysis

The original code scanned the full boid list **three times** per boid
(once per rule), recomputing the same squared distances and square roots
each pass. Here, all three rules accumulate their totals in **one pass**
over neighbors, sharing a single `sqrt` per qualifying pair.

### Structure-of-Arrays Storage

Positions and velocities live in four parallel flat lists
(`positions_x`, `positions_y`, `velocities_x`, `velocities_y`). This
cache-friendly layout avoids Python object attribute lookups in the
hot simulation loop and enables fast bulk operations (reset/addition/
removal).

### Inlined Inner Loop

The grid query and neighbor scan are written directly inside the update
loop rather than factored into per-rule methods, eliminating function
call and intermediate-list overhead. Frequently accessed globals are
hoisted into locals for each frame.

## Tuning

Adjust the constants at the top of `main.py`:

| Constant            | Purpose                                         |
| ------------------- | ----------------------------------------------- |
| `NUM_BOIDS`         | Starting boid count                             |
| `PERCEPTION_RADIUS` | Neighbor detection range (drives grid cell size)|
| `SEPARATION_RADIUS` | Close-range repulsion zone                      |
| `MAX_SPEED`         | Maximum velocity magnitude                      |
| `MAX_FORCE`         | Maximum steering force per frame                |
| `WEIGHT_SEPARATION` | Strength of the separation rule                 |
| `WEIGHT_ALIGNMENT`  | Strength of the alignment rule                  |
| `WEIGHT_COHESION`   | Strength of the cohesion rule                   |

### Performance Notes

- Start at `NUM_BOIDS = 2000` and use `UP`/`DOWN` to find your headroom.
- Higher `PERCEPTION_RADIUS` increases per-query cost and flock cohesion
  but reduces neighbor-locality.
- Raise `MAX_SPEED` / `MAX_FORCE` for more energetic flocks at high counts.
- The rendering cost is O(N) triangle fills and is not the bottleneck for
  counts well into the thousands.
