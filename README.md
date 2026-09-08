# Senior Thesis

A pygame-based boids flocking simulation.

## Overview

This project simulates emergent flocking behavior using [Craig Reynolds'
boids model](https://en.wikipedia.org/wiki/Boids). Each boid independently
combines three local steering rules to produce realistic group motion:

- **Separation** – avoid crowding nearby boids
- **Alignment** – match the average heading of neighbors
- **Cohesion** – steer toward the center of mass of the flock

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

| Key     | Action        |
| ------- | ------------- |
| `SPACE` | Reset the flock                |
| `ESC`   | Quit the simulation            |

## Tuning

The following constants at the top of `main.py` control the simulation:

- `NUM_BOIDS` – number of agents in the flock
- `PERCEPTION_RADIUS` – how far each boid senses others
- `MAX_SPEED` – maximum speed per frame
- `MAX_FORCE` – maximum steering force per frame
- Per-rule weights in `Boid.update()` – adjust balance of separation, alignment, and cohesion
