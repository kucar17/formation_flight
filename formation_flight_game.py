import pygame
import math


class FormationFlightGame:
    def __init__(self):
        self.TB2 = 0
        self.AKINCI = 1
        self.KIZILELMA = 2

        # Initialize Pygame
        # pygame.init()
        # pygame.quit()
        # pygame.display.quit()

        # Render
        self.render = False

        # Screen dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 800, 600
        self.screen = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT + 100)
        )
        pygame.display.set_caption("Formation Flight")

        # Grid settings
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE

        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.DARK_RED = (122, 17, 17)
        self.ORANGE = (255, 165, 0)
        self.BROWN = (139, 69, 19)  # Mountain color
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.GREENISH = (50, 69, 3)
        self.DARK_BLUE = (5, 9, 92)

        # Radar and enemy area parameters
        self.RADAR_CENTER = (400, 300)
        self.RADAR_RADIUS = 175
        self.ENEMY_CENTER = (150, 150)
        self.ENEMY_RADIUS = 75
        self.ALLY_CENTER = (700, 500)
        self.ALLY_RADIUS = 75

        # ALLY AIRCRAFTS CHOICE
        self.AC_TYPES = (self.AKINCI, self.TB2, self.KIZILELMA)

        # Mountain obstacles (example coordinates)
        """MOUNTAINS = [(6, 25), (7, 25), (8, 25), (9, 25), (10, 25),
                    (11, 25), (12, 25), (10, 26), (11, 26), (12, 26),
                    (13, 25), (14, 25), (15, 25)]"""

        self.MOUNTAINS = [
            (4, 25),
            (5, 25),
            (6, 25),
            (7, 25),
            (8, 25),
            (4, 24),
            (5, 24),
            (6, 24),
            (7, 24),
            (8, 24),
            (9, 24),
            (4, 23),
            (5, 23),
            (6, 23),
            (7, 23),
            (8, 23),
            (9, 23),
            (10, 23),
            (11, 23),
        ]

        # Enemy aircraft parameters
        self.ENEMY_AIRCRAFTS = [
            {"angle": 0, "speed": 0.5},
            {"angle": 120, "speed": 0.5},
            {"angle": 240, "speed": 0.5},
        ]
        self.ALLY_AIRCRAFTS = [
            {"type": self.KIZILELMA, "angle": 0, "speed": 0.5},
            {"type": self.TB2, "angle": 120, "speed": 0.5},
            {"type": self.AKINCI, "angle": 240, "speed": 0.5},
        ]

        self.enemy_aircraft_image = pygame.image.load("resources/aircraft.png")
        self.enemy_aircraft_image = pygame.transform.scale(
            self.enemy_aircraft_image, (30, 30)
        )
        self.enemy_aircraft_image = pygame.transform.rotate(
            self.enemy_aircraft_image, 90
        )

        tb2 = pygame.image.load("resources/tb2.png")
        tb2 = pygame.transform.scale(tb2, (65, 65))

        akinci = pygame.image.load("resources/akinci.png")
        akinci = pygame.transform.scale(akinci, (50, 50))

        kizilelma = pygame.image.load("resources/kizilelma.png")
        kizilelma = pygame.transform.scale(kizilelma, (50, 50))
        kizilelma = pygame.transform.rotate(kizilelma, 45)

        self.AIRCRAFTS_SURFACE = [kizilelma, tb2, akinci]

        self.KIZILELMA_INIT_POS = [35, 25]
        self.AKINCI_INIT_POS = [31, 25]
        self.TB2_INIT_POS = [37, 23]

        self.tb2_in_radar = 0
        self.akinci_in_radar = 0
        self.kizilelma_in_radar = 0
        self.tb2_mountain_collide = 0
        self.akinci_mountain_collide = 0
        self.kizilelma_mountain_collide = 0

    def check_mountain_collision(self, cell_x, cell_y, mountain):
        return [cell_x, cell_y] in mountain

    def is_within_radar(self, cell_x, cell_y, radar_center, radar_radius):
        """
        Check if the given grid cell is within the radar area.

        Args:
        cell_x (int): The x-coordinate of the cell.
        cell_y (int): The y-coordinate of the cell.
        radar_center (tuple): A tuple (x, y) representing the center of the radar circle.
        radar_radius (int): The radius of the radar circle.

        Returns:
        bool: True if the cell is within the radar, False otherwise.
        """
        # Calculate the center of the grid cell
        cell_center_x = cell_x + self.CELL_SIZE / 2
        cell_center_y = cell_y + self.CELL_SIZE / 2

        # Calculate distance from the center of the radar to the center of the cell
        distance = math.sqrt(
            (radar_center[0] - cell_center_x) ** 2
            + (radar_center[1] - cell_center_y) ** 2
        )

        # Check if the distance is within the radar radius
        return distance <= radar_radius

    def annotate(
        self,
        tb2_radar_state,
        akinci_radar_state,
        kizilelma_radar_state,
        tb2_radar_point,
        akinci_radar_point,
        kizilelma_radar_point,
        tb2_mountain,
        akinci_mountain,
        kizilelma_mountain,
    ):

        tb2_found = "DETECTED" if tb2_radar_state else "NOT DETECTED"
        akinci_found = "DETECTED" if akinci_radar_state else "NOT DETECTED"
        kizilelma_found = "DETECTED" if kizilelma_radar_state else "NOT DETECTED"

        tb2_collision = "COLLISION" if tb2_mountain else "NO COLLISION"
        akinci_collision = "COLLISION" if akinci_mountain else "NO COLLISION"
        kizilelma_collision = "COLLISION" if kizilelma_mountain else "NO COLLISION"

        font = pygame.font.Font("freesansbold.ttf", 18)
        text = font.render(
            "TB2: "
            + tb2_found
            + " "
            + str(tb2_radar_point)
            + " - "
            + "AKINCI : "
            + akinci_found
            + " "
            + str(akinci_radar_point)
            + " - "
            + "KIZILELMA: "
            + kizilelma_found
            + " "
            + str(kizilelma_radar_point),
            True,
            self.BLACK,
            self.GREENISH,
        )
        textRect = text.get_rect()
        textRect.center = (400, 630)
        self.screen.blit(text, textRect)

        text2 = font.render(
            "TB2: "
            + tb2_collision
            + " - "
            + "AKINCI : "
            + akinci_collision
            + " - "
            + "KIZILELMA: "
            + kizilelma_collision,
            True,
            self.BLACK,
            self.GREENISH,
        )
        textRect2 = text2.get_rect()
        textRect2.center = (400, 670)
        self.screen.blit(text2, textRect2)

    def draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)

    def draw_circle_area(self, center, radius, color):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
                dist = math.sqrt((center[0] - x) ** 2 + (center[1] - y) ** 2)
                if dist <= radius:
                    pygame.draw.rect(
                        self.screen, color, (x, y, self.CELL_SIZE, self.CELL_SIZE)
                    )

    def draw_mountains(self):
        for mountain in self.MOUNTAINS:
            mountain_rect = pygame.Rect(
                mountain[0] * self.CELL_SIZE,
                mountain[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE,
            )
            pygame.draw.rect(self.screen, self.BROWN, mountain_rect)

    def draw_enemy_aircrafts(self):
        for aircraft in self.ENEMY_AIRCRAFTS:
            # Calculate aircraft position based on angle
            x = self.ENEMY_CENTER[0] + math.cos(math.radians(aircraft["angle"])) * (
                self.ENEMY_RADIUS - 25
            )
            y = self.ENEMY_CENTER[1] + math.sin(math.radians(aircraft["angle"])) * (
                self.ENEMY_RADIUS - 25
            )
            # Convert to nearest grid position
            grid_x = int(x) // self.CELL_SIZE * self.CELL_SIZE
            grid_y = int(y) // self.CELL_SIZE * self.CELL_SIZE
            # Draw aircraft as a small rectangle for now
            # pygame.draw.rect(screen, DARK_RED, (grid_x, grid_y, CELL_SIZE, CELL_SIZE))
            self.screen.blit(self.enemy_aircraft_image, (grid_x, grid_y))
            # Update angle for next frame
            aircraft["angle"] += aircraft["speed"]

    def draw_ally_aircrafts(self, selection, state):
        global kizilelma
        global akinci
        global tb2
        index = 0

        for aircraft in self.ALLY_AIRCRAFTS:
            grid_x = state[index][0]
            grid_y = state[index][1]
            # Draw aircraft as a small rectangle for now
            if aircraft["type"] is self.AKINCI:
                akinci = pygame.image.load("resources/akinci.png")
                akinci = pygame.transform.scale(akinci, (50, 50))
                akinci = pygame.transform.rotate(akinci, state[index][2] - 45)
                self.screen.blit(akinci, (grid_x, grid_y))
            elif aircraft["type"] is self.TB2:
                tb2 = pygame.image.load("resources/tb2.png")
                tb2 = pygame.transform.scale(tb2, (60, 60))
                tb2 = pygame.transform.rotate(tb2, state[index][2])
                self.screen.blit(tb2, (grid_x, grid_y))
            else:
                kizilelma = pygame.image.load("resources/kizilelma.png")
                kizilelma = pygame.transform.scale(kizilelma, (50, 50))
                kizilelma = pygame.transform.rotate(kizilelma, state[index][2])
                self.screen.blit(kizilelma, (grid_x, grid_y))

            index += 1

    def calc_radar_time(self, radar_state_prv, radar_state, count):
        if radar_state:
            new_count = count + 1
        else:
            new_count = count

        if radar_state_prv and (not radar_state):
            new_count = 0

        return new_count

    def init(self):
        self.ALLY_POS_PRV = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.KIZILELMA_POS = [
            self.KIZILELMA_INIT_POS[0] * self.CELL_SIZE,
            self.KIZILELMA_INIT_POS[1] * self.CELL_SIZE,
            7,
        ]
        self.AKINCI_POS = [
            self.AKINCI_INIT_POS[0] * self.CELL_SIZE,
            self.AKINCI_INIT_POS[1] * self.CELL_SIZE,
            7,
        ]
        self.TB2_POS = [
            self.TB2_INIT_POS[0] * self.CELL_SIZE,
            self.TB2_INIT_POS[1] * self.CELL_SIZE,
            7,
        ]
        self.ALLY_POS = [self.KIZILELMA_POS, self.TB2_POS, self.AKINCI_POS]

        self.tb2_in_radar_prv = False
        self.akinci_in_radar_prv = False
        self.kizilelma_in_radar_prv = False

        self.tb2_radar_time = 0
        self.akinci_radar_time = 0
        self.kizilelma_radar_time = 0

        self.clock = pygame.time.Clock()

        if self.render:
            self.screen.fill(self.GREENISH)

            self.draw_grid()
            self.draw_circle_area(self.RADAR_CENTER, self.RADAR_RADIUS, self.ORANGE)
            self.draw_circle_area(self.ENEMY_CENTER, self.ENEMY_RADIUS, self.DARK_RED)
            self.draw_circle_area(self.ALLY_CENTER, self.ALLY_RADIUS, self.BLUE)
            self.draw_mountains()
            self.draw_enemy_aircrafts()

    def game_step(self, action):
        if self.render:
            self.screen.fill(self.GREENISH)

            self.draw_grid()
            self.draw_circle_area(self.RADAR_CENTER, self.RADAR_RADIUS, self.ORANGE)
            self.draw_circle_area(self.ENEMY_CENTER, self.ENEMY_RADIUS, self.DARK_RED)
            self.draw_circle_area(self.ALLY_CENTER, self.ALLY_RADIUS, self.BLUE)
            self.draw_mountains()
            self.draw_enemy_aircrafts()

        ## NEW X-Y POSITION ASSIGNMENT
        self.KIZILELMA_POS[0] -= action[0]  # Move x
        self.KIZILELMA_POS[1] -= action[1]  # Move y

        self.TB2_POS[0] -= action[2]
        self.TB2_POS[1] -= action[3]

        self.AKINCI_POS[0] -= action[4]
        self.AKINCI_POS[1] -= action[5]

        # Ensure that aircraft positions do not move outside of the boundaries
        # This checks and corrects any out-of-bounds movement
        self.KIZILELMA_POS[0] = max(
            0, min(self.KIZILELMA_POS[0], self.SCREEN_WIDTH - self.CELL_SIZE)
        )
        self.KIZILELMA_POS[1] = max(
            0, min(self.KIZILELMA_POS[1], self.SCREEN_HEIGHT - self.CELL_SIZE)
        )

        self.TB2_POS[0] = max(
            0, min(self.TB2_POS[0], self.SCREEN_WIDTH - self.CELL_SIZE)
        )
        self.TB2_POS[1] = max(
            0, min(self.TB2_POS[1], self.SCREEN_HEIGHT - self.CELL_SIZE)
        )

        self.AKINCI_POS[0] = max(
            0, min(self.AKINCI_POS[0], self.SCREEN_WIDTH - self.CELL_SIZE)
        )
        self.AKINCI_POS[1] = max(
            0, min(self.AKINCI_POS[1], self.SCREEN_HEIGHT - self.CELL_SIZE)
        )

        self.ALLY_POS = [self.KIZILELMA_POS, self.TB2_POS, self.AKINCI_POS]

        ## HEADING CALCULATION
        self.KIZILELMA_PRV = self.ALLY_POS_PRV[0]
        self.TB2_PRV = self.ALLY_POS_PRV[1]
        self.AKINCI_PRV = self.ALLY_POS_PRV[2]

        self.KIZILELMA_DIFF_Y = self.KIZILELMA_POS[1] - self.KIZILELMA_PRV[1]
        self.KIZILELMA_DIFF_X = self.KIZILELMA_POS[0] - self.KIZILELMA_PRV[0]
        self.KIZILELMA_HEADING = (
            math.atan(self.KIZILELMA_DIFF_Y / self.KIZILELMA_DIFF_X)
            if self.KIZILELMA_DIFF_X != 0
            else math.atan(self.KIZILELMA_DIFF_Y / (self.KIZILELMA_DIFF_X + 0.000001))
        )

        if self.KIZILELMA_HEADING == -0:
            self.KIZILELMA_HEADING = math.pi / 2
        elif self.KIZILELMA_HEADING == 0:
            self.KIZILELMA_HEADING = math.pi / 2

        self.KIZILELMA_POS[2] = self.KIZILELMA_HEADING * 180.0 / math.pi

        self.TB2_DIFF_Y = self.TB2_POS[1] - self.TB2_PRV[1]
        self.TB2_DIFF_X = self.TB2_POS[0] - self.TB2_PRV[0]
        self.TB2_HEADING = (
            math.atan(self.TB2_DIFF_Y / self.TB2_DIFF_X)
            if self.TB2_DIFF_X != 0
            else math.atan(self.TB2_DIFF_Y / (self.TB2_DIFF_X + 0.000001))
        )

        if self.TB2_HEADING == -0:
            self.TB2_HEADING = math.pi / 2
        elif self.TB2_HEADING == 0:
            self.TB2_HEADING = math.pi / 2

        self.TB2_POS[2] = self.TB2_HEADING * 180.0 / math.pi

        self.AKINCI_DIFF_Y = self.AKINCI_POS[1] - self.AKINCI_PRV[1]
        self.AKINCI_DIFF_X = self.AKINCI_POS[0] - self.AKINCI_PRV[0]
        self.AKINCI_HEADING = (
            math.atan(self.AKINCI_DIFF_Y / self.AKINCI_DIFF_X)
            if self.AKINCI_DIFF_X != 0
            else math.atan(self.AKINCI_DIFF_Y / (self.AKINCI_DIFF_X + 0.000001))
        )

        if self.AKINCI_HEADING == -0:
            self.AKINCI_HEADING = math.pi / 2
        elif self.AKINCI_HEADING == 0:
            self.AKINCI_HEADING = math.pi / 2

        self.AKINCI_POS[2] = self.AKINCI_HEADING * 180.0 / math.pi

        self.ALLY_POS = [self.KIZILELMA_POS, self.TB2_POS, self.AKINCI_POS]

        self.tb2_in_radar = self.is_within_radar(
            self.TB2_POS[0], self.TB2_POS[1], self.RADAR_CENTER, self.RADAR_RADIUS
        )
        self.akinci_in_radar = self.is_within_radar(
            self.AKINCI_POS[0],
            self.AKINCI_POS[1],
            self.RADAR_CENTER,
            self.RADAR_RADIUS,
        )
        self.kizilelma_in_radar = self.is_within_radar(
            self.KIZILELMA_POS[0],
            self.KIZILELMA_POS[1],
            self.RADAR_CENTER,
            self.RADAR_RADIUS,
        )

        if self.render:
            for i in self.AC_TYPES:
                self.draw_ally_aircrafts(i, self.ALLY_POS)

        self.tb2_radar_time_new = self.calc_radar_time(
            self.tb2_in_radar_prv, self.tb2_in_radar, self.tb2_radar_time
        )
        self.akinci_radar_time_new = self.calc_radar_time(
            self.akinci_in_radar_prv, self.akinci_in_radar, self.akinci_radar_time
        )
        self.kizilelma_radar_time_new = self.calc_radar_time(
            self.kizilelma_in_radar_prv,
            self.kizilelma_in_radar,
            self.kizilelma_radar_time,
        )

        self.tb2_radar_time = self.tb2_radar_time_new
        self.akinci_radar_time = self.akinci_radar_time_new
        self.kizilelma_radar_time = self.kizilelma_radar_time_new

        self.tb2_mountain_collide = self.check_mountain_collision(
            self.TB2_POS[0], self.TB2_POS[1], self.MOUNTAINS
        )
        self.akinci_mountain_collide = self.check_mountain_collision(
            self.AKINCI_POS[0], self.AKINCI_POS[1], self.MOUNTAINS
        )
        self.kizilelma_mountain_collide = self.check_mountain_collision(
            self.KIZILELMA_POS[0], self.KIZILELMA_POS[1], self.MOUNTAINS
        )

        if self.render:
            self.annotate(
                self.tb2_in_radar,
                self.akinci_in_radar,
                self.kizilelma_in_radar,
                self.tb2_radar_time_new,
                self.akinci_radar_time_new,
                self.kizilelma_radar_time_new,
                self.tb2_mountain_collide,
                self.akinci_mountain_collide,
                self.kizilelma_mountain_collide,
            )
            pygame.display.flip()
        self.clock.tick(30)

        self.ALLY_POS_PRV = [
            [self.ALLY_POS[0][0], self.ALLY_POS[0][1], self.ALLY_POS[0][2]],
            [self.ALLY_POS[1][0], self.ALLY_POS[1][1], self.ALLY_POS[1][2]],
            [self.ALLY_POS[2][0], self.ALLY_POS[2][1], self.ALLY_POS[2][2]],
        ]

        self.tb2_in_radar_prv = self.tb2_radar_time_new
        self.akinci_in_radar_prv = self.akinci_radar_time_new
        self.kizilelma_in_radar_prv = self.kizilelma_radar_time_new
