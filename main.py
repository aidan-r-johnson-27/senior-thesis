# Boids Flocking Simulation
# Implements Craig Reynolds' 1986 boids algorithm for emergent flocking behavior.
# Three simple local rules applied per boid create complex group motion:
#   1. Separation: avoid crowding nearby boids.
#   2. Alignment: match heading of local flockmates.
#   3. Cohesion: steer toward center of local group.
#
# Performance strategy:
# - Spatial hash grid reduces neighbor search from O(n^2) to O(n * k) where k
#   is the average number of boids per cell neighborhood.
# - NumPy vectorization processes all boids in parallel without Python loops.
# - Single-pass neighbor analysis builds query/candidate arrays once, then
#   accumulates separation, alignment, and cohesion in one sort+reduce pass.
# - float32 arrays cut memory bandwidth in half vs float64 (important for the
#   dense pairwise distance calculations).

import math
import sys

import numpy as np
import pygame

# =============================================================================
# Simulation Parameters
# =============================================================================

NUM_BOIDS = 3000                # Starting population; adjustable at runtime with UP/DOWN keys.
PERCEPTION_RADIUS = 50.0        # How far each boid can "see" neighbors.
SEPARATION_RADIUS = 25.0        # Distance at which boids actively avoid collisions.
MAX_SPEED = 4.0                 # Terminal velocity (pixels per frame).
MAX_FORCE = 0.10                # Steering acceleration clamp (prevents jitter at high neighbor counts).

# Rule weights: separation is weighted higher to prevent boids from merging
# into a single point, which would happen if all three were equal.
WEIGHT_SEPARATION = 1.4
WEIGHT_ALIGNMENT = 1.0
WEIGHT_COHESION = 1.0

# Grid cell size = perception radius. This is optimal: smaller cells means
# more neighbor queries but smaller candidate sets; larger cells means fewer
# queries but larger sets. At cell_size = perception_radius, we query exactly
# the 3x3 neighborhood that covers the perception disk.
CELL_SIZE = PERCEPTION_RADIUS

BOID_SIZE = 7                   # Nose-to-tail length in pixels.
WING_ANGLE = 2.35               # Angle from nose to wingtips (~135 degrees from heading).
WING_RATIO = 0.55               # Wing length as fraction of body length.

# Set the limit of polygon rendering. Above this limit render boids
# with fast pixel-dash rendering. The polygon draw loop is 0(n) but still
# hindered by Python. Dash rendering uses NumPy writes and is ~10x faster for
# larger values of n.
TRIANGLE_RENDER_LIMIT = 4500

# =============================================================================
# Pygame Display Setup
# =============================================================================

pygame.init()
# Get the display resolution so we can go fullscreen at native resolution.
_display_info = pygame.display.Info()
WIDTH, HEIGHT = _display_info.current_w, _display_info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Boids Simulation")
clock = pygame.time.Clock()


# =============================================================================
# Derived Constants
# =============================================================================

# We need enough cells to partition the screen so that only
# nearby boids share a cell.
GRID_COLS = int(math.ceil(WIDTH / CELL_SIZE))
GRID_ROWS = int(math.ceil(HEIGHT / CELL_SIZE))
N_CELLS = GRID_COLS * GRID_ROWS

PERC_SQ = PERCEPTION_RADIUS * PERCEPTION_RADIUS
SEP_SQ = SEPARATION_RADIUS * SEPARATION_RADIUS

# Half-world dimensions for toroidal wrapping. If the distance between two
# boids exceeds half the world in any axis, their "shortest" path crosses the edge.
HALF_WIDTH = WIDTH / 2.0
HALF_HEIGHT = HEIGHT / 2.0

# float32 halves bit count per number for pairwise distance calculations.
_DTYPE = np.float32

# Precomputed trig for per-boid triangle vertex math. Each boid is drawn as
# an isosceles triangle: nose at (x + cos*a * size, y + sin*a * size), with
# wingtips rotated by +/- WING_ANGLE from the heading direction.
_COS_WING = math.cos(WING_ANGLE)
_SIN_WING = math.sin(WING_ANGLE)

# =============================================================================
# Boid State Arrays
# =============================================================================

# One (n, 2) array for positions, one for velocities.
# This is faster than an (n, 4) struct-of-arrays because NumPy column slicing
# keeps memory contiguous.
positions = np.zeros((0, 2), dtype=_DTYPE)
velocities = np.zeros((0, 2), dtype=_DTYPE)


def init_boids(count):
    # Create a fresh flock of `count` boids at random positions with random headings.
    # Each boid gets a velocity of magnitude MAX_SPEED pointing in a uniformly
    # random direction. Positions are scattered uniformly across the screen.
    global positions, velocities
    positions = np.random.uniform(0.0, 1.0, (count, 2)).astype(_DTYPE)
    positions[:, 0] *= WIDTH
    positions[:, 1] *= HEIGHT
    angles = np.random.uniform(0.0, 2.0 * math.pi, count)
    velocities = np.column_stack((
        np.cos(angles) * MAX_SPEED,
        np.sin(angles) * MAX_SPEED,
    )).astype(_DTYPE)


def add_boids(count):
    # Append `count` new boids at random positions. Used by the UP key.
    global positions, velocities
    n = positions.shape[0]
    new_pos = np.random.uniform(0.0, 1.0, (count, 2)).astype(_DTYPE)
    new_pos[:, 0] *= WIDTH
    new_pos[:, 1] *= HEIGHT
    angles = np.random.uniform(0.0, 2.0 * math.pi, count)
    new_vel = np.column_stack((
        np.cos(angles) * MAX_SPEED,
        np.sin(angles) * MAX_SPEED,
    )).astype(_DTYPE)
    if n == 0:
        positions, velocities = new_pos, new_vel
    else:
        # Concatenate is fine here since add/remove are infrequent user actions.
        positions = np.concatenate((positions, new_pos), axis=0)
        velocities = np.concatenate((velocities, new_vel), axis=0)


def remove_boids(count):
    # Remove the last `count` boids. Used by the DOWN key.
    global positions, velocities
    n = min(count, positions.shape[0])
    if n > 0:
        # Truncate from the end (cheapest operation; order doesn't matter).
        positions = positions[:-n]
        velocities = velocities[:-n]


def integrate():
    # Apply velocities to positions and clamp speed.
    # This runs after all steering forces have been integrated into velocities.
    # The speed clamp ensures no boid exceeds MAX_SPEED (but can go slower).
    # Toroidal wrapping keeps boids on-screen: positions that go off one edge
    # reappear on the opposite side.
    speed = np.hypot(velocities[:, 0], velocities[:, 1])
    # Avoid division by zero for boids with zero velocity (shouldn't
    # be a worry but who knows).
    speed[speed == 0] = 1.0
    # Scale down velocities that exceed MAX_SPEED. Leave slower ones alone.
    scale = np.minimum(1.0, MAX_SPEED / speed)
    velocities[:, 0] *= scale
    velocities[:, 1] *= scale

    # Update positions. Euler integration (pos += vel).
    positions[:, 0] += velocities[:, 0]
    positions[:, 1] += velocities[:, 1]
    # Wrap around screen edges. % operator always returns a
    # non-negative result when the divisor is positive, which is exactly
    # what we need for toroidal space.
    positions[:, 0] %= WIDTH
    positions[:, 1] %= HEIGHT

# =============================================================================
# Master Step Function
# =============================================================================

def update_all():
    # Run one simulation step: build spatial grid, find neighbors, apply forces.
    # This is the performance-critical path. The algorithm:
    # 1. Assign each boid to a grid cell (counting sort).
    # 2. For each boid, collect candidate neighbors from the 3x3 cell neighborhood.
    # 3. Compute pairwise distances with toroidal wrapping.
    # 4. Filter to actual perception radius.
    # 5. Sort query-candidate pairs by query boid.
    # 6. Reduce neighbor segments to accumulate separation, alignment, cohesion.
    # 7. Apply weighted steering forces and integrate positions.
    n = positions.shape[0]
    if n == 0:
        return

    # ---- Step 1: Spatial hash grid ----
    # Assign each boid to a cell based on its position. I use a counting
    # sort approach: compute cell IDs, count per cell, then build a sorted
    # order array so all boids in the same cell are contiguous.
    cols = (positions[:, 0] / CELL_SIZE).astype(np.int32)
    rows = (positions[:, 1] / CELL_SIZE).astype(np.int32)
    # Clamp to grid bounds (boids exactly on the edge go over).
    cols = np.minimum(cols, GRID_COLS - 1)
    rows = np.minimum(rows, GRID_ROWS - 1)
    # Row-major flattening of the 2D grid.
    cell_ids = rows * GRID_COLS + cols

    # Counting sort: how many boids in each cell?
    counts = np.bincount(cell_ids, minlength=N_CELLS)
    # Prefix sum gives the starting index of each cell in the sorted order.
    starts = np.zeros(N_CELLS + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    # argsort gives the permutation that sorts by cell ID.
    order = np.argsort(cell_ids)

    # ---- Step 2: Gather neighbor pairs ----
    # For each boid, look at the 3x3 block of cells centered on its cell.
    # Toroidal wrapping: (row + dr) % GRID_ROWS handles edges automatically.
    boid_range = np.arange(n, dtype=np.int64)
    q_parts, c_parts = [], []
    for dr in (-1, 0, 1):
        nb_rows = (rows + dr) % GRID_ROWS  # neighbor row with wrapping
        for dc in (-1, 0, 1):
            # Linear IDs of neighbor cells
            nb_cells = nb_rows * GRID_COLS + (cols + dc) % GRID_COLS
            # Where this neighbor's boids start in the sorted order array
            block_start = starts[nb_cells]
            # How many boids are in each neighbor cell
            lens = starts[nb_cells + 1] - block_start
            total = int(lens.sum())
            if total == 0:
                continue
            # block_offset: cumulative offset within each query-boid's
            # contribution to this particular neighbor cell. This lets us
            # compute the exact indices into the `order` array.
            block_offset = np.cumsum(lens) - lens
            # Each query boid repeats `lens[i]` times (one per candidate
            # in that neighbor cell).
            q_parts.append(np.repeat(boid_range, lens))
            # Convert flat candidate indices back to original boid indices.
            lin = np.repeat(block_start, lens) + (
                np.arange(total, dtype=np.int64) - np.repeat(block_offset, lens)
            )
            c_parts.append(order[lin])
    if not q_parts:
        # No neighbors at all (shouldn't happen, but handle gracefully).
        integrate()
        return

    # Concatenate all (query, candidate) pairs from every neighbor cell.
    query = np.concatenate(q_parts)
    cand = np.concatenate(c_parts)
    del q_parts, c_parts

    # ---- Step 3: Pairwise distances with toroidal wrapping ----
    # The world wraps at edges, so the "real" distance between two boids is
    # the minimum of the direct distance and the wrap-around distance.
    # If a component exceeds half the world, it's shorter to go the other way.
    dx = positions[cand, 0] - positions[query, 0]
    dy = positions[cand, 1] - positions[query, 1]
    # Toroidal shortest-vector correction.
    dx -= np.where(dx > HALF_WIDTH, WIDTH, 0.0)
    dx += np.where(dx < -HALF_WIDTH, WIDTH, 0.0)
    dy -= np.where(dy > HALF_HEIGHT, HEIGHT, 0.0)
    dy += np.where(dy < -HALF_HEIGHT, HEIGHT, 0.0)

    # ---- Step 4: Filter to perception radius ----
    d2 = dx * dx + dy * dy  # squared distance (avoid sqrt until needed).
    # Keep pairs within perception radius; exclude self-overlaps (d2 ~ 0).
    mask = (d2 < PERC_SQ) & (d2 > 1e-12)
    query = query[mask]
    cand = cand[mask]
    dx = dx[mask]
    dy = dy[mask]
    d2 = d2[mask]
    del mask
    if query.size == 0:
        integrate()
        return

    # ---- Step 5: Sort by query boid ----
    # Each boid may appear as a query multiple times (once per candidate).
    # Sorting groups all candidates for a given query boid together, which
    # is required for the segment-based reduceat operation.
    query_perm = np.argsort(query, kind="stable")
    query = query[query_perm]
    cand = cand[query_perm]
    dx = dx[query_perm]
    dy = dy[query_perm]
    d2 = d2[query_perm]
    del query_perm

    # ---- Step 6: Segment boundaries ----
    # Consecutive runs of the same query ID define one boid's neighbor list.
    # We use diff + flatnonzero to find where the query ID changes.
    seg_ends = np.flatnonzero(np.diff(query)) + 1
    seg_begins = np.concatenate((np.zeros(1, dtype=np.int64), seg_ends))
    q_uniq = query[seg_begins]  # unique query boid IDs (one per segment)
    # Number of neighbors per boid (segment lengths as float for division).
    neighbor_counts = np.diff(
        np.concatenate((seg_begins, [query.size]))
    ).astype(_DTYPE)

    dist = np.sqrt(d2)
    inv_dist = 1.0 / dist

    # ---- Separation force ----
    # Repel boids that are within SEPARATION_RADIUS. The force direction is
    # the average of unit vectors pointing AWAY from close neighbors. We use
    # reduceat to sum per-segment without an explicit loop.
    # Note the -sign: dx/dy point toward neighbors, so negation points away.
    in_sep = d2 < SEP_SQ
    sep_counts = np.add.reduceat(in_sep, seg_begins).astype(np.float64)
    sep_x_acc = np.add.reduceat(np.where(in_sep, -dx * inv_dist, 0.0), seg_begins)
    sep_y_acc = np.add.reduceat(np.where(in_sep, -dy * inv_dist, 0.0), seg_begins)

    # ---- Alignment & Cohesion ----
    # Both use the same sorted neighbor segments. Alignment sums neighbor
    # velocities; cohesion sums neighbor positions (later subtracted to get
    # the vector toward the center of mass).
    ali_x_acc = np.add.reduceat(velocities[cand, 0], seg_begins)
    ali_y_acc = np.add.reduceat(velocities[cand, 1], seg_begins)
    coh_x_acc = np.add.reduceat(positions[cand, 0], seg_begins)
    coh_y_acc = np.add.reduceat(positions[cand, 1], seg_begins)

    def clamp_force(fx, fy, fmag):
        # Limit steering force magnitude to MAX_FORCE.
        # This prevents boids from accelerating too quickly when surrounded by
        # many neighbors (which would cause oscillation). Forces are rescaled
        # to MAX_FORCE while preserving their direction.
        over = fmag > MAX_FORCE
        fx = np.where(over, fx / np.where(fmag == 0, 1.0, fmag) * MAX_FORCE, fx)
        fy = np.where(over, fy / np.where(fmag == 0, 1.0, fmag) * MAX_FORCE, fy)
        return fx, fy

    # ---- Separation: steer away from close neighbors ----
    # Average the repulsion directions, scale to MAX_SPEED, then subtract
    # current velocity to get the steering correction (Reynolds' steering
    # formula: desired - current).
    sep_x = np.zeros_like(velocities[q_uniq, 0])
    sep_y = np.zeros_like(velocities[q_uniq, 1])
    np.divide(sep_x_acc * MAX_SPEED, sep_counts, out=sep_x, where=sep_counts > 0)
    np.divide(sep_y_acc * MAX_SPEED, sep_counts, out=sep_y, where=sep_counts > 0)
    sep_x = np.where(sep_counts > 0, sep_x - velocities[q_uniq, 0], 0.0)
    sep_y = np.where(sep_counts > 0, sep_y - velocities[q_uniq, 1], 0.0)
    sep_mag = np.hypot(sep_x, sep_y)
    sep_x, sep_y = clamp_force(sep_x, sep_y, sep_mag)

    # ---- Alignment: match neighbor heading ----
    # Desired velocity = average of neighbor velocities, scaled to MAX_SPEED.
    # Steering = desired - current velocity.
    ali_x = (ali_x_acc / neighbor_counts) * MAX_SPEED - velocities[q_uniq, 0]
    ali_y = (ali_y_acc / neighbor_counts) * MAX_SPEED - velocities[q_uniq, 1]
    ali_mag = np.hypot(ali_x, ali_y)
    ali_x, ali_y = clamp_force(ali_x, ali_y, ali_mag)

    # ---- Cohesion: steer toward local center of mass ----
    # Center of mass = mean position of neighbors. We compute the vector
    # from current position toward it, normalize it, scale to MAX_SPEED,
    # then apply Reynolds' steering formula.
    coh_x = coh_x_acc / neighbor_counts - positions[q_uniq, 0]
    coh_y = coh_y_acc / neighbor_counts - positions[q_uniq, 1]
    coh_mag = np.hypot(coh_x, coh_y)
    non_zero = coh_mag > 1e-9
    # Normalize to unit vector, then scale to MAX_SPEED. The where-guard
    # prevents division by zero for boids whose center of mass is exactly
    # at their current position (rare but possible).
    coh_x = np.where(non_zero, coh_x / np.where(coh_mag == 0, 1.0, coh_mag) * MAX_SPEED, 0.0)
    coh_y = np.where(non_zero, coh_y / np.where(coh_mag == 0, 1.0, coh_mag) * MAX_SPEED, 0.0)
    coh_x = np.where(non_zero, coh_x - velocities[q_uniq, 0], 0.0)
    coh_y = np.where(non_zero, coh_y - velocities[q_uniq, 1], 0.0)
    coh_mag = np.hypot(coh_x, coh_y)
    coh_x, coh_y = clamp_force(coh_x, coh_y, coh_mag)

    # ---- Apply weighted forces ----
    # The three steering forces are combined linearly. Weights control the
    # relative importance of each rule (higher weight = stronger influence).
    velocities[q_uniq, 0] += (
        WEIGHT_SEPARATION * sep_x + WEIGHT_ALIGNMENT * ali_x + WEIGHT_COHESION * coh_x
    )
    velocities[q_uniq, 1] += (
        WEIGHT_SEPARATION * sep_y + WEIGHT_ALIGNMENT * ali_y + WEIGHT_COHESION * coh_y
    )

    integrate()


def boid_colors():
    # Compute RGB color for each boid based on its travel direction.
    # Maps the velocity angle (from arctan2) to a hue on the color wheel:
    # - Right    -> Red
    # - Up       -> Green
    # - Left     -> Cyan
    # - Down     -> Blue
    # Returns an (n, 3) uint8 array of RGB values.
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])      
    hue = (angles + math.pi) * (1.0 / (2.0 * math.pi))          

    # HSV-to-RGB conversion: split the hue circle into 6 sectors of 60 degrees each.
    # Within each sector, one channel ramps from 0->255 while another ramps 255->0
    # and the third stays constant. This is faster than a general HSV conversion
    # because we precompute the sector and fractional part once.
    h6 = hue * 6.0
    sector = h6.astype(np.int32) % 6  # which of the 6 color sectors.
    frac = h6 - np.floor(h6)          # position within the sector [0, 1).
    p = np.uint8(0)                    # "black" component (unused but kept for clarity).
    q = (255.0 * (1.0 - frac)).astype(np.uint8)  # ramping down (255 -> 0).
    t = (255.0 * frac).astype(np.uint8)           # ramping up (0 -> 255).
    c255 = np.uint8(255)                          # "white" component.

    # np.select picks the right channel value for each sector. The mapping
    # follows the standard HSV-to-RGB wheel:
    #   Sector 0 (0-60):   R=255, G=t,     B=0    (red -> yellow)
    #   Sector 1 (60-120): R=q,   G=255,   B=0    (yellow -> green)
    #   Sector 2 (120-180):R=0,   G=255,   B=t    (green -> cyan)
    #   Sector 3 (180-240):R=0,   G=q,     B=255  (cyan -> blue)
    #   Sector 4 (240-300):R=t,   G=0,     B=255  (blue -> magenta)
    #   Sector 5 (300-360):R=255, G=0,     B=q    (magenta -> red)
    r = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
                  [c255, q, p, p, t, c255])
    g = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
                  [t, c255, c255, q, p, p])
    b = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
                  [p, p, t, c255, c255, q])
    return np.stack((r, g, b), axis=1)


def draw_boids_tri(surface, colors, cos_a, sin_a, n):
    # Draw each boid as a filled triangle polygon (used for n <= 4500).
    # Each boid is an isosceles triangle with:
    # - Nose vertex: forward along the heading direction by BOID_SIZE pixels
    # - Two wing vertices: rotated +/- WING_ANGLE from the heading, at
    #   WING_RATIO * BOID_SIZE distance from the boid's center
    # The rotation formulas use the standard 2D rotation matrix:
    #     x' = x*cos - y*sin
    #     y' = x*sin + y*cos
    # We precompute cos(WING_ANGLE) and sin(WING_ANGLE) as module constants
    # since they're the same for every boid.
    size = BOID_SIZE
    wing_len = size * WING_RATIO
    cw, sw = _COS_WING, _SIN_WING

    for i in range(n):
        # float() coercion: pygame.draw.polygon rejects numpy.float32 scalars,
        # so we cast to native Python floats. This is a small cost per boid
        # but acceptable for the polygon renderer (used only below 4500 boids).
        x = float(positions[i, 0])
        y = float(positions[i, 1])
        ca = float(cos_a[i])
        sa = float(sin_a[i])

        # Nose: directly ahead of the boid.
        tip_x = x + ca * size
        tip_y = y + sa * size
        # Left wing: rotated by +WING_ANGLE from heading.
        left_x = x + (ca * cw - sa * sw) * wing_len
        left_y = y + (sa * cw + ca * sw) * wing_len
        # Right wing: rotated by -WING_ANGLE from heading.
        right_x = x + (ca * cw + sa * sw) * wing_len
        right_y = y + (sa * cw - ca * sw) * wing_len

        pygame.draw.polygon(surface, (colors[i, 0], colors[i, 1], colors[i, 2]), (
            (tip_x, tip_y),
            (left_x, left_y),
            (right_x, right_y),
        ))


def draw_boids_dash(surface, colors):
    # Draw boids as colored pixels with velocity trails (used for n > 4500).
    # Instead of expensive polygon calls, we write directly to the pixel buffer.
    # Each boid is drawn as a 3-pixel dash: its current position plus two
    # trailing points along its velocity vector. This creates a motion-blur
    # effect that's both faster to render and visually distinct from the
    # triangle mode.
    # The pixel format conversion (RGB to packed native pixel) is necessary
    # because pygame.surfarray.pixels2d() returns the surface's native integer
    # format (which varies by platform: 0xRRGGBB on most, but could differ).
    # pixels2d gives us a 2D view of the raw pixel buffer (surface is locked
    # while this view exists). We write individual pixels directly.
    view = pygame.surfarray.pixels2d(surface)

    # Compute integer pixel positions for current frame and two trail points
    # (trailing by 2x and 4x the velocity vector). np.clip keeps us on-screen.
    ix0 = np.clip(positions[:, 0].astype(np.int32), 0, WIDTH - 1)
    iy0 = np.clip(positions[:, 1].astype(np.int32), 0, HEIGHT - 1)
    ix1 = np.clip((positions[:, 0] + velocities[:, 0] * 2.0).astype(np.int32), 0, WIDTH - 1)
    iy1 = np.clip((positions[:, 1] + velocities[:, 1] * 2.0).astype(np.int32), 0, HEIGHT - 1)
    ix2 = np.clip((positions[:, 0] + velocities[:, 0] * 4.0).astype(np.int32), 0, WIDTH - 1)
    iy2 = np.clip((positions[:, 1] + velocities[:, 1] * 4.0).astype(np.int32), 0, HEIGHT - 1)

    # Pack RGB channels into the surface's native pixel format.
    # Each channel has a "mask" (e.g., 0x00FF0000 for red in ARGB). We need
    # to left-shift each channel value to the correct bit position. The
    # bit position is the index of the lowest set bit in the mask, which we
    # find using (mask & -mask).bit_length() - 1 (isolate lowest bit, count
    # trailing zeros effectively).
    def _shift(mask):
        return (mask & -mask).bit_length() - 1

    mask_r, mask_g, mask_b, _ = surface.get_masks()
    shift_r, shift_g, shift_b = _shift(mask_r), _shift(mask_g), _shift(mask_b)

    # Bitwise-OR the shifted channels together to form packed pixel values.
    rgb = (
        (colors[:, 0].astype(np.int64) << shift_r)
        | (colors[:, 1].astype(np.int64) << shift_g)
        | (colors[:, 2].astype(np.int64) << shift_b)
    ).astype(view.dtype)

    # Write pixels: current position + two trail points.
    view[ix0, iy0] = rgb
    view[ix1, iy1] = rgb
    view[ix2, iy2] = rgb
    del view  # release the surface lock before calling display.flip()


def draw_boids(surface):
    # Dispatch to the appropriate renderer based on boid count.
    # Below TRIANGLE_RENDER_LIMIT: use polygon triangles (prettier, slower).
    # Above it: use pixel-dash rendering (fast, visually distinct).
    # The crossover point balances visual quality against frame rate.
    n = positions.shape[0]
    if n == 0:
        return
    colors = boid_colors()
    if n <= TRIANGLE_RENDER_LIMIT:
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        draw_boids_tri(surface, colors, np.cos(angles), np.sin(angles), n)
    else:
        draw_boids_dash(surface, colors)


def main():
    # Main game loop: event handling, simulation update, rendering.
    # Controls:
    #     SPACE    - Reset all boids to random positions/headings
    #     UP       - Add 100 boids
    #     DOWN     - Remove 100 boids
    #     ESC      - Quit
    init_boids(NUM_BOIDS)
    # Dark blue background for good contrast with the colorful boids.
    background_color = (15, 15, 35)

    running = True
    while running:
        # Process Pygame events (keyboard, window close, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Reset: recreate all boids at current count.
                    init_boids(positions.shape[0])
                elif event.key == pygame.K_UP:
                    add_boids(100)
                elif event.key == pygame.K_DOWN:
                    remove_boids(100)

        # Advance simulation one step (forces + integration).
        update_all()

        # Render frame.
        screen.fill(background_color)
        draw_boids(screen)
        pygame.display.flip()

        # Cap at 60 FPS (clock.tick returns ms since last call)
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
