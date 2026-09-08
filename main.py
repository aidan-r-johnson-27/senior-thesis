import pygame
import sys
import math
import random

# Initialize the pygame engine.
pygame.init()

# --- Simulation configuration -------------------------------------------------
WIDTH, HEIGHT = 800, 600          # Screen dimensions (pixels)
NUM_BOIDS = 100                   # Number of boids in the sim
PERCEPTION_RADIUS = 50            # Distance at which boids begin to notice each other
MAX_SPEED = 3                     # Fastest a boid can travel
MAX_FORCE = 0.1                   # Largest steering force possible per frame

# --- Window setup -------------------------------------------------------------
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boids Simulation")
clock = pygame.time.Clock()


class Boid:
    # A single agent in the flocking simulation.
    # Each boid steers itself by combining three classic flocking rules:
    # separation, alignment, and cohesion.

    def __init__(self):
        # Place the boid at a random position moving in a random direction.
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(angle) * MAX_SPEED
        self.vy = math.sin(angle) * MAX_SPEED

    def update(self, boids):
        # Apply the three steering rules.
        separation_x, separation_y = self.separate(boids)
        alignment_x, alignment_y = self.align(boids)
        cohesion_x, cohesion_y = self.cohere(boids)

        # Weight the contribution of each rule (separation matters most).
        separation_x *= 1.5
        separation_y *= 1.5
        alignment_x *= 1.0
        alignment_y *= 1.0
        cohesion_x *= 1.0
        cohesion_y *= 1.0

        # Integrate the steering forces into velocity.
        self.vx += separation_x + alignment_x + cohesion_x
        self.vy += separation_y + alignment_y + cohesion_y

        # Clamp speed.
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed > MAX_SPEED:
            self.vx = (self.vx / speed) * MAX_SPEED
            self.vy = (self.vy / speed) * MAX_SPEED

        # Move the boid.
        self.x += self.vx
        self.y += self.vy

        # Wrap around the edges of the screen.
        if self.x < 0:
            self.x = WIDTH
        elif self.x > WIDTH:
            self.x = 0
        if self.y < 0:
            self.y = HEIGHT
        elif self.y > HEIGHT:
            self.y = 0

    def separate(self, boids):
        # Steer away from neighbors that are too close.
        steer_x, steer_y = 0, 0
        count = 0
        for other in boids:
            if other is self:
                continue
            dx = self.x - other.x
            dy = self.y - other.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            # Push away with a strength relative to neighbor proximity.
            if dist < PERCEPTION_RADIUS and dist > 0:
                steer_x += dx / dist
                steer_y += dy / dist
                count += 1
        if count > 0:
            steer_x /= count
            steer_y /= count
            # Scale to MAX_SPEED and limit by MAX_FORCE.
            speed = math.sqrt(steer_x ** 2 + steer_y ** 2)
            if speed > 0:
                steer_x = (steer_x / speed) * MAX_SPEED - self.vx
                steer_y = (steer_y / speed) * MAX_SPEED - self.vy
                force = math.sqrt(steer_x ** 2 + steer_y ** 2)
                if force > MAX_FORCE:
                    steer_x = (steer_x / force) * MAX_FORCE
                    steer_y = (steer_y / force) * MAX_FORCE
        return steer_x, steer_y

    def align(self, boids):
        # Point toward the average heading of nearby boids.
        avg_vx, avg_vy = 0, 0
        count = 0
        for other in boids:
            if other is self:
                continue
            dx = self.x - other.x
            dy = self.y - other.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < PERCEPTION_RADIUS:
                avg_vx += other.vx
                avg_vy += other.vy
                count += 1
        if count > 0:
            avg_vx /= count
            avg_vy /= count
            # Normalize the average velocity to MAX_SPEED, then take the delta from our own velocity and clamp it.
            speed = math.sqrt(avg_vx ** 2 + avg_vy ** 2)
            if speed > 0:
                avg_vx = (avg_vx / speed) * MAX_SPEED
                avg_vy = (avg_vy / speed) * MAX_SPEED
            steer_x = avg_vx - self.vx
            steer_y = avg_vy - self.vy
            force = math.sqrt(steer_x ** 2 + steer_y ** 2)
            if force > MAX_FORCE:
                steer_x = (steer_x / force) * MAX_FORCE
                steer_y = (steer_y / force) * MAX_FORCE
            return steer_x, steer_y
        return 0, 0

    def cohere(self, boids):
        # Steer toward the center of mass of nearby boids.
        avg_x, avg_y = 0, 0
        count = 0
        for other in boids:
            if other is self:
                continue
            dx = self.x - other.x
            dy = self.y - other.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < PERCEPTION_RADIUS:
                avg_x += other.x
                avg_y += other.y
                count += 1
        if count > 0:
            avg_x /= count
            avg_y /= count
            dx = avg_x - self.x
            dy = avg_y - self.y
            speed = math.sqrt(dx ** 2 + dy ** 2)
            if speed > 0:
                dx = (dx / speed) * MAX_SPEED
                dy = (dy / speed) * MAX_SPEED
            steer_x = dx - self.vx
            steer_y = dy - self.vy
            force = math.sqrt(steer_x ** 2 + steer_y ** 2)
            if force > MAX_FORCE:
                steer_x = (steer_x / force) * MAX_FORCE
                steer_y = (steer_y / force) * MAX_FORCE
            return steer_x, steer_y
        return 0, 0

    def draw(self, surface):
        # Render the boid as a small triangle pointing in its travel direction.
        angle = math.atan2(self.vy, self.vx)
        length = 10
        tip_x = self.x + math.cos(angle) * length
        tip_y = self.y + math.sin(angle) * length
        left_x = self.x + math.cos(angle + 2.5) * length * 0.6
        left_y = self.y + math.sin(angle + 2.5) * length * 0.6
        right_x = self.x + math.cos(angle - 2.5) * length * 0.6
        right_y = self.y + math.sin(angle - 2.5) * length * 0.6
        pygame.draw.polygon(surface, (0, 0, 0), [
            (tip_x, tip_y),
            (left_x, left_y),
            (right_x, right_y)
        ])

# Create the initial flock.
boids = [Boid() for _ in range(NUM_BOIDS)]

# --- Main loop ----------------------------------------------------------------
while True:
    # Handle user inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_SPACE:
                boids = [Boid() for _ in range(NUM_BOIDS)]

    # Clear the background.
    screen.fill((255, 255, 255))

    # Update every boid based on the flock, then draw them.
    for boid in boids:
        boid.update(boids)
    for boid in boids:
        boid.draw(screen)

    # Present the frame and keep the simulation at 60 FPS.
    pygame.display.flip()
    clock.tick(60)
