""" Implements the logic of the Flappy Bird game.
"""
import numpy as np


from enum import IntEnum

# Constants
BALL_SPEED = 1
PADDLE_SPEED = 3
MAX_SCORE = 10


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Ball:
    def __init__(self, x, y, vel_x, vel_y):
        self.x = x
        self.y = y
        self.vel_x = vel_x
        self.vel_y = vel_y


class Actions(IntEnum):
    """ Possible actions for the player to take. """
    IDLE_1, MOVE_UP_1, MOVE_DOWN_1, IDLE_2, MOVE_UP_2, MOVE_DOWN_2 = 0, 1, 2, 3, 4, 5


def is_collide(x1, y1, x2, y2, r):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2 <= r ** 2


class TennisLogic:
    """ Handles the logic of the Tennis game.

    The implementation of this class is decoupled from the implementation of the
    game's graphics. This class implements the logical portion of the game.

    Args:
        screen_size (Tuple[int, int]): Tuple with the screen's width and height.
        paddle_size (int): Size of players' paddles.

    Attributes:
        player1 (Player): The 1st player's position.
        player2 (Player): The 2nd player's position.
        ball (Ball): The ball's position
        score1 (int): The 1st player's current score.
        score2 (int): The 2nd player's current score
    """

    def __init__(self,
                 screen_size,
                 paddle_size,
                 ball_radius):
        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]
        self._paddle_size = paddle_size
        self._ball_radius = ball_radius

        self.player1 = None
        self.player2 = None
        self.ball = None

        self.score1 = 0
        self.score2 = 0

        self.set_initial_positions()

    def set_initial_positions(self):
        """ Sets ball to the middle and assign random velocity,
        places player's paddles to default positions."""
        self.player1 = Player(30, self._screen_height - 60)
        self.player2 = Player(self._screen_width - 30, 60)

        vel_x = 0.5 + np.random.rand() / 2  # to make sure it is not too vertical
        vel_x *= np.random.choice([-1, 1])
        vel_y = np.sqrt(1 - vel_x ** 2)
        vel_y *= np.random.choice([-1, 1])
        self.ball = Ball(self._screen_width // 2,
                         self._screen_height // 2,
                         BALL_SPEED * vel_x,
                         BALL_SPEED * vel_y)

    def update_state(self,
                     action: Actions):
        """ Updates board according to the action taken

        Args:
            action (Union[Actions]): The action taken by players.

        Returns:
            `1` if player1 scored
            `-1` if player2 scored and
            `0` if no one scored."""
        # update ball's position
        self.ball.x += self.ball.vel_x
        self.ball.y += self.ball.vel_y

        # update players' positions
        if action == Actions.MOVE_UP_1 \
                and self.player1.y > self._paddle_size:
            self.player1.y -= PADDLE_SPEED
        elif action == Actions.MOVE_DOWN_1 \
                and self.player1.y < self._screen_height:
            self.player1.y += PADDLE_SPEED
        elif action == Actions.MOVE_UP_2 \
                and self.player2.y > self._paddle_size:
            self.player2.y -= PADDLE_SPEED
        elif action == Actions.MOVE_DOWN_2 \
                and self.player2.y < self._screen_height:
            self.player2.y += PADDLE_SPEED

        hit = self.check_hit()
        scored = self.check_score()
        return scored, hit

    def check_hit(self):
        """ Checks if ball has hit players' paddles or top and bottom wall"""
        hit = 0
        # Player1's paddle edge and ball hit
        if is_collide(self.player1.x, self.player1.y,
                      self.ball.x, self.ball.y, self._ball_radius)\
                and self.ball.vel_x < 0:
            self.ball.vel_x *= -1
            self.ball.vel_y *= -1
            hit = 1

        if is_collide(self.player1.x, self.player1.y - self._paddle_size,
                      self.ball.x, self.ball.y, self._ball_radius)\
                and self.ball.vel_x < 0:
            self.ball.vel_x *= -1
            self.ball.vel_y *= -1
            hit = 1

        # Player2's paddle edge and ball hit
        if is_collide(self.player2.x, self.player2.y,
                      self.ball.x, self.ball.y, self._ball_radius)\
                and self.ball.vel_x > 0:
            self.ball.vel_x *= -1
            self.ball.vel_y *= -1
            hit = -1

        if is_collide(self.player2.x, self.player2.y - self._paddle_size,
                      self.ball.x, self.ball.y, self._ball_radius)\
                and self.ball.vel_x > 0:
            self.ball.vel_x *= -1
            self.ball.vel_y *= -1
            hit = -1

        # Player1 paddle and ball hit
        if 0 <= self.ball.x - self.player1.x <= self._ball_radius \
                and -self._ball_radius <= self.player1.y - self.ball.y <= self._paddle_size + self._ball_radius\
                and self.ball.vel_x < 0:
            self.ball.vel_x *= -1
            hit = 1

        # Player2 paddle and ball hit
        if 0 < self.player2.x - self.ball.x <= self._ball_radius \
                and -self._ball_radius <= self.player2.y - self.ball.y <= self._paddle_size + self._ball_radius\
                and self.ball.vel_x > 0:
            self.ball.vel_x *= -1
            hit = -1

        # Top wall and ball hit
        if self.ball.y >= self._screen_height - self._ball_radius:
            self.ball.vel_y *= -1

        # Bottom wall and ball hit
        if self.ball.y <= self._ball_radius:
            self.ball.vel_y *= -1

        return hit

    def check_score(self):
        """ Check if anyone score

        Returns:
            `1` if player1 scored
            `-1` if player2 scored and
            `0` if no one scored."""
        scored = 0

        if self.ball.x <= 0:
            self.score2 += 1
            scored = -1
            self.set_initial_positions()

        elif self.ball.x >= self._screen_width:
            self.score1 += 1
            scored = 1
            self.set_initial_positions()

        return scored
