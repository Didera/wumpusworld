import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pygame
import sys
from collections import deque
import os

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 100)
# Initialize pygame
pygame.init()
pygame.display.set_mode((1, 1))
# Load images with alpha for transparency
AGENT_IMG = pygame.image.load("agent.png").convert_alpha()
WUMPUS_IMG = pygame.image.load("wumpus.png").convert_alpha()
PIT_IMG = pygame.image.load("pit.png").convert_alpha()
GOLD_IMG = pygame.image.load("gold.png").convert_alpha()

# --------------------------
# 1. Wumpus World Definition
# --------------------------
class WumpusWorld:
    def __init__(self, world_template=None):
        self.grid_size = 4
        self.agent_pos = (0, 0)
        self.agent_dir = "right"
        self.has_gold = False
        self.has_arrow = True
        self.wumpus_alive = True
        self.world = self.clone_world(world_template) if world_template else self.generate_world()
        self.percepts = self.get_percepts()

    def clone_world(self, template):
        import copy
        return copy.deepcopy(template)

    def generate_world(self):
        world = [[{"pit": False, "wumpus": False, "gold": False} for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place exactly one pit (not at start)
        pits_placed = 0
        while pits_placed < 2:
            i, j = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (i, j) not in [(0, 0), (0, 1), (1, 0)] and not world[i][j]["pit"]:
                world[i][j]["pit"] = True
                pits_placed += 1

        # Place Wumpus (not on pit or start)
        while True:
            x, y = random.randint(0, 3), random.randint(0, 3)
            if (x, y) != (0, 0) and not world[x][y]["pit"]:
                world[x][y]["wumpus"] = True
                break

        # Place Gold (not on pit, wumpus, or start)
        while True:
            x, y = random.randint(0, 3), random.randint(0, 3)
            if (x, y) != (0, 0) and not world[x][y]["pit"] and not world[x][y]["wumpus"]:
                world[x][y]["gold"] = True
                break

        return world

    def get_percepts(self):
        x, y = self.agent_pos
        cell = self.world[x][y]
        percepts = {"stench": False, "breeze": False, "glitter": False, "bump": False, "scream": False}
        
        # Check adjacent cells for stench and breeze
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.world[nx][ny]["wumpus"] and self.wumpus_alive:
                    percepts["stench"] = True
                if self.world[nx][ny]["pit"]:
                    percepts["breeze"] = True
        
        if cell["gold"]:
            percepts["glitter"] = True
        return percepts

    def move_forward(self):
        x, y = self.agent_pos
        if self.agent_dir == "up":
            x -= 1
        elif self.agent_dir == "down":
            x += 1
        elif self.agent_dir == "left":
            y -= 1
        elif self.agent_dir == "right":
            y += 1
        
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.agent_pos = (x, y)
            self.percepts = self.get_percepts()
            return True
        else:
            self.percepts["bump"] = True
            return False

    def turn_left(self):
        directions = ["up", "left", "down", "right"]
        current_idx = directions.index(self.agent_dir)
        next_idx = (current_idx + 1) % 4
        self.animate_rotation(self.agent_dir, directions[next_idx])
        self.agent_dir = directions[next_idx]

    def turn_right(self):
        directions = ["up", "right", "down", "left"]
        current_idx = directions.index(self.agent_dir)
        next_idx = (current_idx + 1) % 4
        self.animate_rotation(self.agent_dir, directions[next_idx])
        self.agent_dir = directions[next_idx]


    def grab_gold(self):
        x, y = self.agent_pos
        if self.world[x][y]["gold"]:
            self.has_gold = True
            self.world[x][y]["gold"] = False
            self.percepts["glitter"] = False
            return True
        return False

    def shoot_arrow(self):
        if not self.has_arrow:
            return False

        self.has_arrow = False
        x, y = self.agent_pos
        direction = self.agent_dir
        wumpus_killed = False

        # Arrow movement in the current facing direction
        if direction == "up":
            for i in range(x - 1, -1, -1):
                if self.world[i][y]["wumpus"]:
                    wumpus_killed = True
                    self.world[i][y]["wumpus"] = False
                    break
        elif direction == "down":
            for i in range(x + 1, self.grid_size):
                if self.world[i][y]["wumpus"]:
                    wumpus_killed = True
                    self.world[i][y]["wumpus"] = False
                    break
        elif direction == "left":
            for j in range(y - 1, -1, -1):
                if self.world[x][j]["wumpus"]:
                    wumpus_killed = True
                    self.world[x][j]["wumpus"] = False
                    break
        elif direction == "right":
            for j in range(y + 1, self.grid_size):
                if self.world[x][j]["wumpus"]:
                    wumpus_killed = True
                    self.world[x][j]["wumpus"] = False
                    break

        if wumpus_killed:
            self.wumpus_alive = False
            self.percepts["scream"] = True
            print("-----------Wumpus killed! You hear a scream.")
            self.percepts = self.get_percepts()
            return True
        else:
            print("--------------Arrow missed the Wumpus.")
            return False

    def is_game_over(self):
        x, y = self.agent_pos
        if self.world[x][y]["pit"]:
            return "lose_pit"
        if self.world[x][y]["wumpus"] and self.wumpus_alive:
            return "lose_wumpus"
        if self.has_gold and self.agent_pos == (0, 0):
            return "win"
        return "continue"
    
    def animate_rotation(self, from_dir, to_dir):
        direction_order = ["up", "right", "down", "left"]
        from_index = direction_order.index(from_dir)
        to_index = direction_order.index(to_dir)

        steps = (to_index - from_index) % 4
        if steps == 3:
            steps = -1  # Rotate left one step instead of 3 right

        for _ in range(abs(steps)):
            if steps > 0:
                self.agent_dir = direction_order[(direction_order.index(self.agent_dir) + 1) % 4]
            else:
                self.agent_dir = direction_order[(direction_order.index(self.agent_dir) - 1) % 4]

            visualize_world(self, agent)  # Update drawing
            pygame.time.delay(200)        # 200 ms Delay to simulate animation

# ---------------------
# 2. Improved Logic Agent
# ---------------------
class LogicAgent:
    def __init__(self, env, knowledge=None):
        self.visit_counter = {}
        self.env = env
        self.kb_safe = {(0, 0)}
        self.kb_visited = {(0, 0)}
        self.kb_danger = set()
        self.kb_wumpus = set()
        self.kb_pit = set()
        self.path_history = [(0, 0)]
        self.step_count = 0
        
        if knowledge:
            knowledge.apply_to_agent(self)

    def get_neighbors(self, pos):
        x, y = pos
        return [(nx, ny) for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
                if 0 <= nx < self.env.grid_size and 0 <= ny < self.env.grid_size]

    def update_knowledge(self, pos, percepts):
        """Update knowledge base based on current position and percepts"""
        neighbors = self.get_neighbors(pos)
        
        # Mark current position as visited and safe
        self.kb_visited.add(pos)
        self.kb_safe.add(pos)
        self.kb_danger.discard(pos)
        
        if not percepts["breeze"] and not percepts["stench"]:
            #all neighbors are safe
            for neighbor in neighbors:
                self.kb_safe.add(neighbor)
                self.kb_danger.discard(neighbor)
                self.kb_pit.discard(neighbor)
                self.kb_wumpus.discard(neighbor)
        else:
            #mark unvisited neighbors as potentially dangerous
            if percepts["breeze"]:
                for neighbor in neighbors:
                    if neighbor not in self.kb_visited and neighbor not in self.kb_safe:
                        self.kb_danger.add(neighbor)
                        self.kb_pit.add(neighbor)
            
            if percepts["stench"]:
                for neighbor in neighbors:
                    if neighbor not in self.kb_visited and neighbor not in self.kb_safe:
                        self.kb_danger.add(neighbor)
                        self.kb_wumpus.add(neighbor)

    def get_safe_moves(self, current_pos):
        """Get list of safe moves from current position"""
        neighbors = self.get_neighbors(current_pos)
        safe_unvisited = [n for n in neighbors if n in self.kb_safe and n not in self.kb_visited]
        return safe_unvisited

    def get_risky_moves(self, current_pos, risk_threshold=0.6):
        """Get list of potentially risky but possible moves"""
        neighbors = self.get_neighbors(current_pos)
        unknown_neighbors = [n for n in neighbors 
                           if n not in self.kb_visited 
                           and n not in self.kb_safe 
                           and n not in self.kb_danger]
        
        if unknown_neighbors and random.random() < risk_threshold:
            return unknown_neighbors
        return []

    def get_backtrack_moves(self, current_pos):
        """Get safe visited neighbors for backtracking"""
        neighbors = self.get_neighbors(current_pos)
        safe_visited = [n for n in neighbors if n in self.kb_safe and n in self.kb_visited]
        return safe_visited

    def find_path_to_start(self):
        """Find path back to (0,0) using BFS through safe cells"""
        if self.env.agent_pos == (0, 0):
            return []
        
        queue = deque([(self.env.agent_pos, [])])
        visited = {self.env.agent_pos}
        
        while queue:
            current, path = queue.popleft()
            
            if current == (0, 0):
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor in self.kb_safe:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []

    def move_to_position(self, target_pos):
        """Move agent to target position (assuming it's adjacent)"""
        current_x, current_y = self.env.agent_pos
        target_x, target_y = target_pos

        # Determine required direction
        if target_x < current_x:
            required_dir = "up"
        elif target_x > current_x:
            required_dir = "down"
        elif target_y < current_y:
            required_dir = "left"
        elif target_y > current_y:
            required_dir = "right"
        else:
            return False  # Same position
        
        # Turn to face the required direction
        while self.env.agent_dir != required_dir:
            self.env.turn_right()
        
        # Move forward
        if self.env.move_forward():
            self.kb_visited.add(target_pos)
            self.path_history.append(target_pos)
            self.visit_counter[target_pos] = self.visit_counter.get(target_pos, 0) + 1


            return True
        return False

    def shoot_at_suspected_wumpus(self):
        """Try to shoot wumpus in suspected locations"""
        if not self.env.has_arrow:
            return False
        
        current_pos = self.env.agent_pos
        
        # Check each direction for suspected wumpus
        directions = ["up", "down", "left", "right"]
        
        for direction in directions:
            # Turn to face this direction
            while self.env.agent_dir != direction:
                self.env.turn_right()
            
            # Check if there's a suspected wumpus in this direction
            x, y = current_pos
            if direction == "up":
                cells_in_line = [(x-i, y) for i in range(1, self.env.grid_size) if x-i >= 0]
            elif direction == "down":
                cells_in_line = [(x+i, y) for i in range(1, self.env.grid_size) if x+i < self.env.grid_size]
            elif direction == "left":
                cells_in_line = [(x, y-i) for i in range(1, self.env.grid_size) if y-i >= 0]
            elif direction == "right":
                cells_in_line = [(x, y+i) for i in range(1, self.env.grid_size) if y+i < self.env.grid_size]
            
            # If any cell in line is suspected to have wumpus, shoot
            for cell in cells_in_line:
                if cell in self.kb_wumpus:
                    print(f"🎯 Shooting arrow {direction} at suspected wumpus location {cell}")
                    success = self.env.shoot_arrow()
                    if success:
                        # Remove wumpus from suspected locations
                        self.kb_wumpus.clear()
                        # Update knowledge - cells that were dangerous due to wumpus are now safe
                        for wumpus_cell in list(self.kb_danger):
                            if wumpus_cell in self.kb_wumpus:
                                self.kb_danger.discard(wumpus_cell)
                                self.kb_safe.add(wumpus_cell)
                    return success
        
        return False

    def execute_strategy(self):
    
        print("==========Starting Wumpus World exploration...")

        while True:
            current_pos = self.env.agent_pos
            percepts = self.env.percepts
            self.step_count += 1

            print(f"\n---------Step {self.step_count} at {current_pos}")
            print(f"   Percepts: {percepts}")
            print(f"   Safe cells: {self.kb_safe}")
            print(f"   Dangerous cells: {self.kb_danger}")

            # Count how many times agent has encountered breeze/stench
            if percepts["breeze"] or percepts["stench"]:
                self.percept_counter = getattr(self, "percept_counter", 0) + 1
            else:
                self.percept_counter = getattr(self, "percept_counter", 0)

            # If agent gets stuck seeing breeze/stench > 10 times, jump to a risky cell
            if self.percept_counter > 10:
                print("//////////////Repeated breeze/stench over 10 times. Jumping to a risky cell...")
                danger_neighbors = [n for n in self.get_neighbors(current_pos) if n in self.kb_danger]
                if danger_neighbors:
                    target = random.choice(danger_neighbors)
                    print(f"///////////////Jumping into danger at {target}")
                    if self.move_to_position(target):
                        self.percept_counter = 0  # reset after jump
                        continue

            # Check for immediate danger
            game_status = self.env.is_game_over()
            if game_status == "lose_pit":
                print("-------------Fell into a pit! Game over.")
                return "lose"
            elif game_status == "lose_wumpus":
                print("------------------Eaten by Wumpus! Game over.")
                return "lose"
            elif game_status == "win":
                print("-----------------Victory! Returned with gold!")
                return "win"

            # Check if gold is found
            if percepts["glitter"]:
                print("------------------Found gold! Grabbing it...")
                self.env.grab_gold()
                print("--------------------Returning to start...")
                return self.return_to_start()

            # Update knowledge base
            self.update_knowledge(current_pos, percepts)

            # Try to shoot if stench detected
            if percepts["stench"] and self.env.has_arrow:
                if self.shoot_at_suspected_wumpus():
                    print("---------------- Wumpus eliminated! Updating knowledge...")
                    self.update_knowledge(current_pos, self.env.get_percepts())

            moved = False 

            #Deterministic Unsafe Move If No Alternatives
            if not moved:
                danger_neighbors = [n for n in agent.get_neighbors(current_pos) if n in agent.kb_danger]
                safe_or_unknown_neighbors = [n for n in agent.get_neighbors(current_pos)
                                            if n not in agent.kb_danger]

                if not safe_or_unknown_neighbors and danger_neighbors:
                    # Agent is trapped by danger — choose the least visited dangerous cell
                    target = min(danger_neighbors, key=lambda n: agent.visit_counter.get(n, 0))
                    print(f"---------------------Trapped! Deliberately entering dangerous cell: {target}")
                    agent.move_to_position(target)
                    moved = True
            #Safe move
            safe_moves = self.get_safe_moves(current_pos)
            if safe_moves:
                target = random.choice(safe_moves)
                print(f"---------------Moving to safe cell: {target}")
                if self.move_to_position(target):
                    moved = True
                    continue

            #Risky move
            risky_moves = self.get_risky_moves(current_pos, risk_threshold=0.4)
            if risky_moves:
                target = random.choice(risky_moves)
                print(f"----------------------Taking calculated risk, moving to: {target}")
                if self.move_to_position(target):
                    moved = True
                    continue

            #Backtrack
            backtrack_moves = self.get_backtrack_moves(current_pos)
            if backtrack_moves:
                target = random.choice(backtrack_moves)
                print(f"------------------------Backtracking to: {target}")
                if self.move_to_position(target):
                    moved = True
                    continue

            #If visited 3+ times, go into known danger
            if not moved:
                visit_count = self.visit_counter.get(current_pos, 0)
                danger_neighbors = [n for n in self.get_neighbors(current_pos) if n in self.kb_danger]
                if visit_count >= 3 and danger_neighbors:
                    if random.random() < 1.0:  # Always take risk after 3 visits
                        target = random.choice(danger_neighbors)
                        print(f"--------------------Been here {visit_count} times! Risking move to: {target}")
                        if self.move_to_position(target):
                            moved = True
                            continue

            #Last resort move
            if not moved:
                all_neighbors = self.get_neighbors(current_pos)
                available_moves = [n for n in all_neighbors if n != current_pos]
                if available_moves:
                    target = random.choice(available_moves)
                    print(f"-------------------Last resort move to: {target}")
                    if self.move_to_position(target):
                        moved = True
                        continue

            # Nothing worked
            if not moved:
                print("🚫 No moves possible - this shouldn't happen!")
                return "stuck"


    def return_to_start(self):
        """Return to starting position with gold"""
        print("🏃 Navigating back to start...")
        path_to_start = self.find_path_to_start()
        
        if not path_to_start:
            print("---------------- No safe path to start found!")
            return "lose"
        
        for step in path_to_start:
            if not self.move_to_position(step):
                print(f"--------------- Failed to move to {step}")
                return "lose"
            print(f"---------------Moved to {step}")
        
        if self.env.agent_pos == (0, 0) and self.env.has_gold:
            print("--------------------- Successfully returned to start with gold!")
            return "win"
        else:
            print("-----------------Failed to return to start properly.")
            return "lose"

# ----------------------
# 3. Visualization Code
# ----------------------


GRID_SIZE = 4
CELL_SIZE = 100
PADDING = 40
STATUS_WIDTH = 220
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 2 * PADDING + STATUS_WIDTH
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 2 * PADDING
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption(" Wumpus World")
font = pygame.font.SysFont("consolas", 22)

# Define colors
WHITE = (245, 245, 245)
BLACK = (15, 15, 15)
SAFE_GREEN = (80, 220, 100)
DANGER_RED = (255, 120, 120)
VISITED_GRAY = (200, 200, 200)
AGENT_BLUE = (50, 150, 255)
BORDER_COLOR = (20, 20, 20)
BG_COLOR = (30, 30, 60)
TEXT_COLOR = (220, 220, 220)

#images
AGENT_IMG = pygame.transform.scale(pygame.image.load("agent.png").convert_alpha(), (50, 50))
WUMPUS_IMG = pygame.transform.scale(pygame.image.load("wumpus.png").convert_alpha(), (50, 50))
PIT_IMG = pygame.transform.scale(pygame.image.load("pit.png").convert_alpha(), (50, 50))
GOLD_IMG = pygame.transform.scale(pygame.image.load("gold.png").convert_alpha(), (50, 50))

def draw_cell(x, y, color, label=""):
    rect = pygame.Rect(PADDING + y * CELL_SIZE, PADDING + (GRID_SIZE - 1 - x) * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, color, rect, border_radius=12)
    pygame.draw.rect(screen, BORDER_COLOR, rect, 2, border_radius=12)
    if label:
        txt = font.render(label, True, BLACK)
        screen.blit(txt, txt.get_rect(center=rect.center))

def draw_agent(x, y, direction):
    center_x = PADDING + y * CELL_SIZE + CELL_SIZE // 2
    center_y = PADDING + (GRID_SIZE - 1 - x) * CELL_SIZE + CELL_SIZE // 2
    angle_map = {
        "right": 0,
        "down": 90,
        "left": 180,
        "up": 270
    }
    angle = angle_map.get(direction, 0)
    rotated_img = pygame.transform.rotate(AGENT_IMG, angle)
    rect = rotated_img.get_rect(center=(center_x, center_y))
    screen.blit(rotated_img, rect)

def highlight_cell(x, y, color):
    rect = pygame.Rect(PADDING + y * CELL_SIZE, PADDING + (GRID_SIZE - 1 - x) * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, color, rect, width=5, border_radius=10)

def draw_status_panel(agent, env):
    base_x = GRID_SIZE * CELL_SIZE + 2 * PADDING + 10
    status_texts = [
        f"Pos: {env.agent_pos}",
        f"Dir: {env.agent_dir}",
        f"Gold: {'YES' if env.has_gold else 'NO'}",
        f"Arrows: {1 if env.has_arrow else 0}",
        f"Steps: {agent.step_count}"
    ]
    for i, text in enumerate(status_texts):
        txt_surface = font.render(text, True, TEXT_COLOR)
        screen.blit(txt_surface, (base_x, 50 + i * 40))


def visualize_world(env, agent):
    screen.fill(BG_COLOR)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            cell = (x, y)
            color = WHITE

            if cell in agent.kb_safe:
                color = SAFE_GREEN
            elif cell in agent.kb_danger:
                color = DANGER_RED
            elif cell in agent.kb_visited:
                color = VISITED_GRAY

            draw_cell(x, y, color)

            img_rect = pygame.Rect(
                PADDING + y * CELL_SIZE + (CELL_SIZE - 50) // 2,
                PADDING + (GRID_SIZE - 1 - x) * CELL_SIZE + (CELL_SIZE - 50) // 2,
                50, 50
            )

            if env.world[x][y]["pit"]:
                screen.blit(PIT_IMG, img_rect)
            if env.world[x][y]["wumpus"] and env.wumpus_alive:
                screen.blit(WUMPUS_IMG, img_rect)
            if env.world[x][y]["gold"]:
                screen.blit(GOLD_IMG, img_rect)

    # Highlight agent position
    ax, ay = env.agent_pos
    highlight_cell(ax, ay, AGENT_BLUE)
    draw_agent(ax, ay, env.agent_dir)
    draw_status_panel(agent, env)

    pygame.display.update()

# ----------------------
# 4. Knowledge Base
# ----------------------

class KnowledgeBase:
    def __init__(self):
        self.learned_safe = set()
        self.learned_danger = set()
        self.learned_pits = set()
        self.learned_wumpus = set()

    def update_from_agent(self, agent):
        self.learned_safe.update(agent.kb_safe)
        self.learned_danger.update(agent.kb_danger)
        self.learned_pits.update(agent.kb_pit)
        self.learned_wumpus.update(agent.kb_wumpus)

    def apply_to_agent(self, agent):
        agent.kb_safe = self.learned_safe.copy()
        agent.kb_danger = self.learned_danger.copy()
        agent.kb_pit = self.learned_pits.copy()
        agent.kb_wumpus = self.learned_wumpus.copy()
        agent.kb_visited = {(0, 0)}  # Always start fresh for visited cells

# -----------------
# 5. Main Execution
# -----------------
if __name__ == "__main__":
    print("----------------------------Fixed Wumpus World Logic Agent Starting...")
    
    # Generate the world layout
    temp_env = WumpusWorld()
    saved_world = temp_env.world
    print("----World layout generated:")
    for i in range(4):
        for j in range(4):
            cell = saved_world[i][j]
            contents = []
            if cell["pit"]: contents.append("P")
            if cell["wumpus"]: contents.append("W") 
            if cell["gold"]: contents.append("G")
            print(f"({i},{j}): {contents if contents else 'Empty'}")
    
    # Initialize shared knowledge base
    shared_knowledge = KnowledgeBase()
    
    # Running the agent
    env = WumpusWorld(world_template=saved_world)
    agent = LogicAgent(env, knowledge=shared_knowledge)
    
    #visualization
    clock = pygame.time.Clock()
    running = True
    auto_play = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if auto_play:
            #current state visualization
            visualize_world(env, agent)
            
            # Execute one step manually for visualization
            current_pos = env.agent_pos
            percepts = env.percepts
            agent.step_count += 1
            
            print(f"\n...................Step {agent.step_count} at {current_pos}")
            print(f"   Percepts: {percepts}")
            
            # Check for immediate danger first
            game_status = env.is_game_over()
            if game_status == "lose_pit":
                print("..................Fell into a pit! Game over.")
                auto_play = False
                continue
            elif game_status == "lose_wumpus":
                print("...................Eaten by Wumpus! Game over.")
                auto_play = False
                continue
            elif game_status == "win":
                print(".....................Victory! Returned with gold!")
                auto_play = False
                continue
            
            # Check if gold is found
            # If gold is found, grab it and return to start
            if percepts["glitter"]:
                print(".........................Found gold! Grabbing it...")
                env.grab_gold()
                print(".........................Returning to start...")
                # Execute return to start
                path_to_start = agent.find_path_to_start()
                for step in path_to_start:
                    agent.move_to_position(step)
                    visualize_world(env, agent)
                    pygame.time.delay(500)
                auto_play = False
                continue
            
            # Update knowledge
            agent.update_knowledge(current_pos, percepts)
            
            # Try to shoot wumpus if detected
            if percepts["stench"] and env.has_arrow:
                if agent.shoot_at_suspected_wumpus():
                    print("...........................Wumpus eliminated!")
                    agent.update_knowledge(current_pos, env.get_percepts())
            
            # Execute movement strategy
            moved = False
            #Deterministic Unsafe Move If No Alternatives
            if not moved:
                danger_neighbors = [n for n in agent.get_neighbors(current_pos) if n in agent.kb_danger]
                safe_or_unknown_neighbors = [n for n in agent.get_neighbors(current_pos)
                                            if n not in agent.kb_danger]

                if not safe_or_unknown_neighbors and danger_neighbors:
                    # Agent is trapped by danger — choose the least visited dangerous cell
                    target = min(danger_neighbors, key=lambda n: agent.visit_counter.get(n, 0))
                    print(f"...........................Trapped! Deliberately entering dangerous cell: {target}")
                    agent.move_to_position(target)
                    moved = True
            #Safe moves
            safe_moves = agent.get_safe_moves(current_pos)
            if safe_moves and not moved:
                target = random.choice(safe_moves)
                print(f"-----Moving to safe cell: {target}")
                if agent.move_to_position(target):
                    moved = True
            
            # Risky moves
            
            if not moved:
                risk_threshold = 1 if agent.step_count >= 25 else 0.8
                risky_moves = agent.get_risky_moves(current_pos, risk_threshold=risk_threshold)
                if risky_moves:
                    target = random.choice(risky_moves)
                    print(f"------Prioritized risk move after 30+ steps, moving to: {target}" if agent.step_count >= 30 else f"====Taking calculated risk, moving to: {target}")
                    if agent.move_to_position(target):
                        moved = True
            

            #Backtrack
            if not moved:
                neighbors = agent.get_neighbors(current_pos)
                danger_neighbors = [n for n in neighbors if n in agent.kb_danger]
                backtrack_moves = agent.get_backtrack_moves(current_pos)

                # Check if completely surrounded (no safe or risky options)
                surrounded = all(n in agent.kb_danger or n in agent.kb_visited for n in neighbors)

                if surrounded:
                    if backtrack_moves:
                        target = random.choice(backtrack_moves)
                        print(f"-----Surrounded — safely backtracking to {target}")
                        if agent.move_to_position(target):
                            moved = True
                    elif danger_neighbors:
                        if random.random() < 0.7:  # 70% chance to risk
                            target = random.choice(danger_neighbors)
                            print(f"--------Surrounded — forced risky move into: {target}")
                            if agent.move_to_position(target):
                                moved = True


            #Last resort - any move
            if not moved:
                all_neighbors = agent.get_neighbors(current_pos)
                available_moves = [n for n in all_neighbors if n != current_pos]
                if available_moves:
                    target = random.choice(available_moves)
                    print(f"------Last resort move to: {target}")
                    agent.move_to_position(target)
                    moved = True
            
            if not moved:
                print("------No moves possible!")
                auto_play = False
            
            clock.tick(1)  
        else:
           
            visualize_world(env, agent)
            clock.tick(10)
    
    pygame.quit()
    print("Game finished!")