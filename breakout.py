import numpy as np
import random

class BreakoutEnv:
    def __init__(self, width=15, height=10, layout_name="default"):
        self.width = width
        self.height = height

        # Paddle settings
        self.paddle_width = 5
        self.paddle_y = height - 1
        self.paddle_x = (width - self.paddle_width) // 2
        self.paddle_speed = 0
        self.paddle_max_speed = 2

        # Ball settings
        self.ball_x = None
        self.ball_y = None
        self.ball_vx = None
        self.ball_vy = 1

        # Bricks
        self.layout_name = layout_name
        self.brick_layout = self.get_layout(layout_name)
        self.bricks = set(self.brick_layout)

        self.reset()

    def get_layout(self, name):
        if name == "rectangle":
            return self.rectangle_layout()
        elif name == "triangle":
            return self.triangle_layout()
        elif name == "zigzag":
            return self.zigzag_layout()
        else:
            return self.default_brick_layout()

    def default_brick_layout(self):
        return [(i * 3, 1) for i in range(5)]

    def rectangle_layout(self):
        return [(x * 3, y) for y in range(2) for x in range(4)]

    def triangle_layout(self):
        layout = []
        for y in range(3):
            start_x = (self.width - (y + 1) * 3) // 2
            for x in range(y + 1):
                layout.append((start_x + x * 3, y))
        return layout

    def zigzag_layout(self):
        return [(i * 3, i % 2) for i in range(5)]

    def reset(self):
        # Reset paddle
        self.paddle_x = (self.width - self.paddle_width) // 2
        self.paddle_speed = 0

        # Reset ball
        self.ball_x = self.paddle_x + self.paddle_width // 2
        self.ball_y = self.paddle_y - 1
        self.ball_vx = random.choice([-2, -1, 0, 1, 2])
        self.ball_vy = 1

        # Reset bricks
        self.bricks = set(self.brick_layout)

        return self.get_state()

    def step(self, action):
        # Action is -1, 0 or +1
        self.paddle_speed = max(-self.paddle_max_speed, min(self.paddle_max_speed, self.paddle_speed + action))
        self.paddle_x = max(0, min(self.width - self.paddle_width, self.paddle_x + self.paddle_speed))

        # Update ball position
        next_x = self.ball_x + self.ball_vx
        next_y = self.ball_y + self.ball_vy

        # Handle wall collisions
        if next_x < 0 or next_x >= self.width:
            self.ball_vx *= -1
        if next_y < 0:
            self.ball_vy *= -1

        # Update ball after wall bounce
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        reward = -1  # Penalize each time step

        # Check for brick hit
        for brick in list(self.bricks):
            bx, by = brick
            if bx <= self.ball_x < bx + 3 and self.ball_y == by:
                self.bricks.remove(brick)
                self.ball_vy *= -1
                break

        # Check for paddle hit
        if self.ball_y == self.paddle_y and self.paddle_x <= self.ball_x < self.paddle_x + self.paddle_width:
            index = self.ball_x - self.paddle_x
            self.ball_vx = [-2, -1, 0, 1, 2][index]
            self.ball_vy = -1

        # Check for ball missed
        if self.ball_y >= self.height:
            self.reset()

        done = len(self.bricks) == 0
        return self.get_state(), reward, done

    def get_state(self):
        return (self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
                self.paddle_x, self.paddle_speed, tuple(sorted(self.bricks)))

    def render(self):
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        for (bx, by) in self.bricks:
            for i in range(3):
                if 0 <= bx + i < self.width:
                    grid[by][bx + i] = "#"

        for i in range(self.paddle_width):
            px = self.paddle_x + i
            if 0 <= px < self.width:
                grid[self.paddle_y][px] = "="

        if 0 <= self.ball_y < self.height and 0 <= self.ball_x < self.width:
            grid[self.ball_y][self.ball_x] = "O"

        print("\n".join("".join(row) for row in grid))
        print("-" * self.width)
