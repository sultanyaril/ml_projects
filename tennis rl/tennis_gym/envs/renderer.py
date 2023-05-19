""" Implements the game's renderer, responsible from drawing the game on the
screen.
"""

import pygame


class TennisRenderer:
    def __init__(self,
                 screen_size,
                 paddle_size,
                 ball_radius,
                 FPS):
        pygame.init()
        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]
        self._paddle_size = paddle_size
        self._ball_radius = ball_radius
        self._FPS = FPS

        self.display = None
        self.game = None
        self.surface = pygame.Surface(screen_size)
        self._clock = pygame.time.Clock()

    def make_display(self) -> None:
        """ Initializes the pygame's display.

        Required for drawing images on the screen.
        """
        self.display = pygame.display.set_mode((self._screen_width,
                                                self._screen_height))

    def draw_surface(self, show_score: bool = True) -> None:
        """ Re-draws the renderer's surface."""
        self.surface.fill('black')
        pygame.draw.circle(self.surface, 'white', (self.game.ball.x, self.game.ball.y), self._ball_radius)
        pygame.draw.line(self.surface, 'red',
                         (self.game.player1.x, self.game.player1.y),
                         (self.game.player1.x, self.game.player1.y - self._paddle_size))
        pygame.draw.line(self.surface, 'blue',
                         (self.game.player2.x, self.game.player2.y),
                         (self.game.player2.x, self.game.player2.y - self._paddle_size))
        self.display.blit(self.surface, [0, 0])

    def draw_score(self):
        font = pygame.font.Font(None, 24)
        text_surface = font.render("{}:{}".format(self.game.score1, self.game.score2),
                                   True, "white")
        text_width, _ = text_surface.get_size()
        center_x = self._screen_width // 2 - text_width // 2
        center_y = 24
        self.display.blit(text_surface, (center_x, center_y))

    def update_display(self):
        """ Updates the display with the current surface of the renderer."""
        self.draw_surface()
        self.draw_score()
        pygame.display.update()
        self._clock.tick(self._FPS)

