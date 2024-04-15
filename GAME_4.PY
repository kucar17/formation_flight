import pygame
import sys
import math

TB2       = 0
AKINCI    = 1
KIZILELMA = 2

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 100))
pygame.display.set_caption('Formation Flight')

# Grid settings
CELL_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
DARK_RED = (122, 17, 17)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)  # Mountain color
GREEN = (0, 255, 0)
BLUE  = (0, 0, 255)
BLACK = (0, 0, 0)
GREENISH = (50, 69, 3)
DARK_BLUE = (5, 9, 92)

# Radar and enemy area parameters
RADAR_CENTER = (400, 300)
RADAR_RADIUS = 175
ENEMY_CENTER = (80, 80)
ENEMY_RADIUS = 75
ALLY_CENTER  = (700, 500)
ALLY_RADIUS  = 75

# ALLY AIRCRAFTS CHOICE
AC_TYPES = (AKINCI, TB2, KIZILELMA)

# Mountain obstacles (example coordinates)
"""MOUNTAINS = [(6, 25), (7, 25), (8, 25), (9, 25), (10, 25), 
             (11, 25), (12, 25), (10, 26), (11, 26), (12, 26), 
             (13, 25), (14, 25), (15, 25)]"""

MOUNTAINS = [(4, 25), (5, 25), (6, 25),
             (7, 25), (8, 25), 
             (4, 24), (5, 24), (6, 24), 
             (7, 24), (8, 24), (9, 24),
             (4, 23), (5, 23), (6, 23),
             (7, 23), (8, 23), (9, 23), (10, 23), (11, 23)]

# Enemy aircraft parameters
ENEMY_AIRCRAFTS = [{'angle': 0, 'speed': 0.5}, {'angle': 120, 'speed': 0.5}, {'angle': 240, 'speed': 0.5}]
ALLY_AIRCRAFTS  = [{'type': KIZILELMA, 'angle': 0, 'speed': 0.5}, {'type': TB2, 'angle': 120, 'speed': 0.5}, {'type': AKINCI, 'angle': 240, 'speed': 0.5}]

# Load aircraft image
enemy_aircraft_image = pygame.image.load("aircraft.png")
enemy_aircraft_image = pygame.transform.scale(enemy_aircraft_image, (30, 30))
enemy_aircraft_image = pygame.transform.rotate(enemy_aircraft_image, 90)

tb2  = pygame.image.load("tb2.png")
tb2 = pygame.transform.scale(tb2, (65, 65))

akinci  = pygame.image.load("akinci.png")
akinci = pygame.transform.scale(akinci, (50, 50))

kizilelma  = pygame.image.load("kizilelma.png")
kizilelma = pygame.transform.scale(kizilelma, (50, 50))
kizilelma = pygame.transform.rotate(kizilelma, 45)

AIRCRAFTS_SURFACE = [kizilelma, tb2, akinci]

KIZILELMA_INIT_POS = [35, 25]
AKINCI_INIT_POS    = [31, 25]
TB2_INIT_POS       = [37, 23]

def is_within_radar(cell_x, cell_y, radar_center, radar_radius):
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
    cell_center_x = cell_x + CELL_SIZE / 2
    cell_center_y = cell_y + CELL_SIZE / 2
    
    # Calculate distance from the center of the radar to the center of the cell
    distance = math.sqrt((radar_center[0] - cell_center_x) ** 2 + (radar_center[1] - cell_center_y) ** 2)
    
    # Check if the distance is within the radar radius
    return distance <= radar_radius

def annotate(tb2_radar_state, akinci_radar_state, kizilelma_radar_state, 
             tb2_radar_point, akinci_radar_point, kizilelma_radar_point):

    tb2_found       = "DETECTED" if tb2_radar_state       else "NOT DETECTED"
    akinci_found    = "DETECTED" if akinci_radar_state    else "NOT DETECTED"
    kizilelma_found = "DETECTED" if kizilelma_radar_state else "NOT DETECTED"

    font     = pygame.font.Font('freesansbold.ttf', 18)
    text     = font.render('TB2: ' + tb2_found + " " + str(tb2_radar_point) + " - " + "AKINCI : " + akinci_found + " " + str(akinci_radar_point) +  " - " + "KIZILELMA: " + kizilelma_found + " " + str(kizilelma_radar_point), True, BLACK, GREENISH)
    textRect = text.get_rect()
    textRect.center = (400, 650)
    screen.blit(text, textRect)

def draw_grid():
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

def draw_circle_area(center, radius, color):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            dist = math.sqrt((center[0] - x)**2 + (center[1] - y)**2)
            if dist <= radius:
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

def draw_mountains():
    for mountain in MOUNTAINS:
        mountain_rect = pygame.Rect(mountain[0] * CELL_SIZE, mountain[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BROWN, mountain_rect)

def draw_enemy_aircrafts():
    for aircraft in ENEMY_AIRCRAFTS:
        # Calculate aircraft position based on angle
        x = ENEMY_CENTER[0] + math.cos(math.radians(aircraft['angle'])) * (ENEMY_RADIUS - 25)
        y = ENEMY_CENTER[1] + math.sin(math.radians(aircraft['angle'])) * (ENEMY_RADIUS - 25)
        # Convert to nearest grid position
        grid_x = int(x) // CELL_SIZE * CELL_SIZE
        grid_y = int(y) // CELL_SIZE * CELL_SIZE
        # Draw aircraft as a small rectangle for now
        #pygame.draw.rect(screen, DARK_RED, (grid_x, grid_y, CELL_SIZE, CELL_SIZE))
        screen.blit(enemy_aircraft_image, (grid_x, grid_y))
        # Update angle for next frame
        aircraft['angle'] += aircraft['speed']

def draw_ally_aircrafts(selection, state):
    global kizilelma
    global akinci
    global tb2
    index = 0

    for aircraft in ALLY_AIRCRAFTS:
        grid_x = state[index][0] 
        grid_y = state[index][1]
        # Draw aircraft as a small rectangle for now
        if aircraft['type'] is AKINCI:
            akinci  = pygame.image.load("akinci.png")
            akinci = pygame.transform.scale(akinci, (50, 50))
            akinci = pygame.transform.rotate(akinci, state[index][2] - 45)
            screen.blit(akinci, (grid_x, grid_y))
        elif aircraft['type'] is TB2:
            tb2  = pygame.image.load("tb2.png")
            tb2 = pygame.transform.scale(tb2, (60, 60))
            tb2 = pygame.transform.rotate(tb2, state[index][2])
            screen.blit(tb2, (grid_x, grid_y))
        else:
            kizilelma  = pygame.image.load("kizilelma.png")
            kizilelma = pygame.transform.scale(kizilelma, (50, 50))
            kizilelma = pygame.transform.rotate(kizilelma, state[index][2])
            screen.blit(kizilelma, (grid_x, grid_y))       

        index += 1

def calc_radar_time(radar_state_prv, radar_state, count):
    if (radar_state):
        new_count = count + 1
    else:
        new_count = count

    if (radar_state_prv and (not radar_state)):
        new_count = 0

    return new_count

def main():    

    ALLY_POS_PRV  = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    #TB2_POS       = [20 * CELL_SIZE, 25 * CELL_SIZE, 90]
    #AKINCI_POS    = [30 * CELL_SIZE, 20 * CELL_SIZE, 50]
    KIZILELMA_POS = [KIZILELMA_INIT_POS[0] * CELL_SIZE, KIZILELMA_INIT_POS[1] * CELL_SIZE, 7]
    AKINCI_POS    = [AKINCI_INIT_POS[0]    * CELL_SIZE, AKINCI_INIT_POS[1]    * CELL_SIZE, 7]
    TB2_POS       = [TB2_INIT_POS[0]       * CELL_SIZE, TB2_INIT_POS[1]       * CELL_SIZE, 7]
    ALLY_POS      = [KIZILELMA_POS, TB2_POS, AKINCI_POS]

    tb2_in_radar_prv       = False
    akinci_in_radar_prv    = False
    kizilelma_in_radar_prv = False

    tb2_radar_time       = 0
    akinci_radar_time    = 0
    kizilelma_radar_time = 0

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(GREENISH)
        
        draw_grid()
        draw_circle_area(RADAR_CENTER, RADAR_RADIUS, ORANGE)
        draw_circle_area(ENEMY_CENTER, ENEMY_RADIUS, DARK_RED)
        draw_circle_area(ALLY_CENTER, ALLY_RADIUS, BLUE)
        draw_mountains()
        draw_enemy_aircrafts()

        ## NEW X-Y POSITION ASSIGNMENT
        KIZILELMA_POS[0] = KIZILELMA_POS[0] - 1 if KIZILELMA_POS[0] - 1 >= 0 else 0 
        KIZILELMA_POS[1] = KIZILELMA_POS[1] - 1 if KIZILELMA_POS[1] - 1 >= 0 else 0 

        TB2_POS[0] = TB2_POS[0] - 1 if TB2_POS[0] - 1 >= 0 else 0 
        TB2_POS[1] = TB2_POS[1] - 1 if TB2_POS[1] - 1 >= 0 else 0 

        AKINCI_POS[0] = AKINCI_POS[0] - 1 if AKINCI_POS[0] - 1 >= 0 else 0 
        AKINCI_POS[1] = AKINCI_POS[1] - 1 if AKINCI_POS[1] - 1 >= 0 else 0 

        ALLY_POS         = [KIZILELMA_POS, TB2_POS, AKINCI_POS]

        ## HEADING CALCULATION
        KIZILELMA_PRV     = ALLY_POS_PRV[0]
        TB2_PRV           = ALLY_POS_PRV[1]
        AKINCI_PRV        = ALLY_POS_PRV[2]

        KIZILELMA_DIFF_Y  = KIZILELMA_POS[1] - KIZILELMA_PRV[1]
        KIZILELMA_DIFF_X  = KIZILELMA_POS[0] - KIZILELMA_PRV[0]
        KIZILELMA_HEADING = math.atan(KIZILELMA_DIFF_Y / KIZILELMA_DIFF_X) if KIZILELMA_DIFF_X != 0 else math.atan(KIZILELMA_DIFF_Y / (KIZILELMA_DIFF_X + 0.000001))

        if (KIZILELMA_HEADING == -0):
            KIZILELMA_HEADING = math.pi / 2
        elif (KIZILELMA_HEADING == 0):
            KIZILELMA_HEADING = math.pi / 2

        KIZILELMA_POS[2]  = KIZILELMA_HEADING * 180.0 / math.pi

        TB2_DIFF_Y  = TB2_POS[1] - TB2_PRV[1]
        TB2_DIFF_X  = TB2_POS[0] - TB2_PRV[0]
        TB2_HEADING = math.atan(TB2_DIFF_Y / TB2_DIFF_X) if TB2_DIFF_X != 0 else math.atan(TB2_DIFF_Y / (TB2_DIFF_X + 0.000001))

        if (TB2_HEADING == -0):
            TB2_HEADING = math.pi / 2
        elif (TB2_HEADING == 0):
            TB2_HEADING = math.pi / 2

        TB2_POS[2]  = TB2_HEADING * 180.0 / math.pi

        AKINCI_DIFF_Y  = AKINCI_POS[1] - AKINCI_PRV[1]
        AKINCI_DIFF_X  = AKINCI_POS[0] - AKINCI_PRV[0]
        AKINCI_HEADING = math.atan(AKINCI_DIFF_Y / AKINCI_DIFF_X) if AKINCI_DIFF_X != 0 else math.atan(AKINCI_DIFF_Y / (AKINCI_DIFF_X + 0.000001))

        if (AKINCI_HEADING == -0):
            AKINCI_HEADING = math.pi / 2
        elif (AKINCI_HEADING == 0):
            AKINCI_HEADING = math.pi / 2

        AKINCI_POS[2]  = AKINCI_HEADING * 180.0 / math.pi

        ALLY_POS          = [KIZILELMA_POS, TB2_POS, AKINCI_POS]   

        tb2_in_radar       = is_within_radar(TB2_POS[0]      , TB2_POS[1]      , RADAR_CENTER, RADAR_RADIUS)
        akinci_in_radar    = is_within_radar(AKINCI_POS[0]   , AKINCI_POS[1]   , RADAR_CENTER, RADAR_RADIUS)
        kizilelma_in_radar = is_within_radar(KIZILELMA_POS[0], KIZILELMA_POS[1], RADAR_CENTER, RADAR_RADIUS)

        for i in AC_TYPES:
            draw_ally_aircrafts(i, ALLY_POS)

        tb2_radar_time_new       = calc_radar_time(tb2_in_radar_prv      , tb2_in_radar      , tb2_radar_time      ) 
        akinci_radar_time_new    = calc_radar_time(akinci_in_radar_prv   , akinci_in_radar   , akinci_radar_time   ) 
        kizilelma_radar_time_new = calc_radar_time(kizilelma_in_radar_prv, kizilelma_in_radar, kizilelma_radar_time) 

        tb2_radar_time       = tb2_radar_time_new
        akinci_radar_time    = akinci_radar_time_new
        kizilelma_radar_time = kizilelma_radar_time_new


        #KIZILELMA_POS[1] -= 1
        annotate(tb2_in_radar, akinci_in_radar, kizilelma_in_radar, 
                 tb2_radar_time_new, akinci_radar_time_new, kizilelma_radar_time_new)
        pygame.display.flip()
        clock.tick(30)

        ALLY_POS_PRV = [[ALLY_POS[0][0], ALLY_POS[0][1], ALLY_POS[0][2]], [ALLY_POS[1][0], ALLY_POS[1][1], ALLY_POS[1][2]],
                        [ALLY_POS[2][0], ALLY_POS[2][1], ALLY_POS[2][2]]]
        
        tb2_in_radar_prv       = tb2_radar_time_new
        akinci_in_radar_prv    = akinci_radar_time_new
        kizilelma_in_radar_prv = kizilelma_radar_time_new

               

if __name__ == '__main__':
    main()