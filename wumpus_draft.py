import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
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

        # Random pits (except start)
        # Place exactly one pit
        while True:
            i, j = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (i, j) != (0, 0):
                world[i][j]["pit"] = True
                break

        # Place Wumpus — ensure not on pit or start
        while True:
            x, y = random.randint(0, 3), random.randint(0, 3)
            if (x, y) != (0, 0) and not world[x][y]["pit"]:
                world[x][y]["wumpus"] = True
                break

        # Place Gold — ensure not on pit or Wumpus or start
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
        else:
            self.percepts["bump"] = True

    def turn_left(self):
        dirs = ["up", "left", "down", "right"]
        self.agent_dir = dirs[(dirs.index(self.agent_dir) + 1) % 4]

    def turn_right(self):
        dirs = ["up", "right", "down", "left"]
        self.agent_dir = dirs[(dirs.index(self.agent_dir) + 1) % 4]

    def grab_gold(self):
        x, y = self.agent_pos
        if self.world[x][y]["gold"]:
            self.has_gold = True
            self.world[x][y]["gold"] = False
            self.percepts["glitter"] = False

    def shoot_arrow(self):
        if not self.has_arrow:
            return False

        self.has_arrow = False
        x, y = self.agent_pos
        dir = self.agent_dir
        wumpus_killed = False

        # Arrow travels in a straight line in the current direction
        if dir == "up":
            for i in range(x - 1, -1, -1):
                if self.world[i][y]["wumpus"]:
                    wumpus_killed = True
                    self.world[i][y]["wumpus"] = False  # Remove Wumpus
                    break
        elif dir == "down":
            for i in range(x + 1, self.grid_size):
                if self.world[i][y]["wumpus"]:
                    wumpus_killed = True
                    self.world[i][y]["wumpus"] = False  # Remove Wumpus
                    break
        elif dir == "left":
            for j in range(y - 1, -1, -1):
                if self.world[x][j]["wumpus"]:
                    wumpus_killed = True
                    self.world[x][j]["wumpus"] = False  # Remove Wumpus
                    break
        elif dir == "right":
            for j in range(y + 1, self.grid_size):
                if self.world[x][j]["wumpus"]:
                    wumpus_killed = True
                    self.world[x][j]["wumpus"] = False  # Remove Wumpus
                    break

        if wumpus_killed:
            self.wumpus_alive = False
            self.percepts["scream"] = True
            print("💥 Wumpus shot! You hear a scream.")
            self.percepts = self.get_percepts()  # Recalculate percepts to remove stench
            return True

        # 👇 Arrow missed — fallback behavior: backup 1 tile
        print("🏹 Arrow missed. Backing up...")

        opposite_moves = {
            "up": (x + 1, y),
            "down": (x - 1, y),
            "left": (x, y + 1),
            "right": (x, y - 1)
        }

        back_pos = opposite_moves.get(self.agent_dir)
        if back_pos:
            bx, by = back_pos
            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                if not self.world[bx][by]["pit"]:
                    print(f"↩️ Backed up to {back_pos} after missing.")
                    self.agent_pos = back_pos
                    self.percepts = self.get_percepts()
                else:
                    print(f"⚠️ Backup cell {back_pos} has a pit. Staying put.")
            else:
                print("⛔ Backup move would go out of bounds. Staying put.")
        return False

    def is_game_over(self):
        x, y = self.agent_pos
        if self.world[x][y]["pit"] or (self.world[x][y]["wumpus"] and self.wumpus_alive):
            return "lose"
        if self.has_gold and self.agent_pos == (0, 0):
            return "win"
        return "continue"

# ---------------------
# 2. Logic Agent Class
# ---------------------
class LogicAgent:
    
    def __init__(self, env, knowledge=None):
        self.env = env
        self.kb_safe = {(0, 0)}
        self.kb_visited = {(0, 0)}
        self.kb_danger = set()
        self.path = [(0, 0)]

        if knowledge:
            knowledge.apply_to_agent(self)

    def update_kb(self, pos, percepts):
        neighbors = self.get_neighbors(pos)

        # Mark the current cell as visited and safe
        self.kb_visited.add(pos)
        self.kb_safe.add(pos)

        # Remove the current cell from dangerous possibilities (if wrongly marked)
        self.kb_danger.discard(pos)

        if not percepts["breeze"] and not percepts["stench"]:
            # If no danger is perceived, mark all neighbors as safe
            for cell in neighbors:
                self.kb_safe.add(cell)
                self.kb_danger.discard(cell)
        else:
            # If breeze or stench is present, cautiously flag unvisited + unknown neighbors
            for cell in neighbors:
                if (cell not in self.kb_safe) and (cell not in self.kb_visited):
                    self.kb_danger.add(cell)

    def deduce_and_act(self, risk_threshold=0.5):
        while True:
            percepts = self.env.percepts
            curr = self.env.agent_pos
            print(f"🧠 Step at {curr} with percepts {percepts}")

            # 🎯 Grab gold if here
            if percepts["glitter"]:
                print("✨ Glitter detected! Grabbing gold...")
                self.env.grab_gold()
                print("🎉 Gold grabbed! Backtracking to exit.")
                self.backtrack()
                return "win"

            # 🧠 Update knowledge
            self.update_kb(curr, percepts)

            # 🔫 If stench and has arrow, shoot
            if percepts["stench"] and self.env.has_arrow:
                print("💨 Stench detected. Trying to shoot Wumpus...")
                shot = self.env.shoot_arrow()
                self.env.percepts = self.env.get_percepts()

                if not shot:
                    print("❌ Missed Wumpus. Trying risky move or backtracking.")
                    risky_next = self.get_safe_unvisited(curr, risk_threshold)
                    if risky_next:
                        self.move_to(risky_next[0])
                        continue
                    else:
                        self.backtrack()
                        continue

            # 💨 If breeze (near pit): don't stop, take calculated risk
            if percepts["breeze"]:
                print("🌬️ Breeze detected. Pit nearby! Calculating risk...")
                risky_next = self.get_safe_unvisited(curr, risk_threshold)
                if risky_next:
                    self.move_to(risky_next[0])
                else:
                    print("🔁 No risky option found. Backtracking.")
                    self.backtrack()
                continue

            # 📍 Normal move
            next_cells = self.get_safe_unvisited(curr, risk_threshold)
            if next_cells:
                self.move_to(next_cells[0])
            else:
                print("🛑 No moves left. Backtracking as last resort.")
                self.backtrack()

            # 🛑 Check game status
            status = self.env.is_game_over()
            if status == "lose":
                print("💀 Agent fell into a pit or got eaten. RIP.")
                return "lose"
            elif status == "win":
                print("🏆 Agent returned safely with the gold!")
                return "win"


            
    def deduce_safe_moves(self):
        safe_moves = []

        # Find cells that are definitely safe
        for cell in self.kb_safe:
            if cell not in self.kb_visited:
                safe_moves.append(cell)

        # Advanced deduction: if we have constraints, try to deduce more
        for cell in self.kb_visited:
            neighbors = self.get_neighbors(cell)
            unknown_neighbors = [n for n in neighbors if n not in self.kb_visited]

            if len(unknown_neighbors) == 1:
                unknown_cell = unknown_neighbors[0]

                # Check if this unknown cell must be dangerous
                if unknown_cell in self.kb_danger:
                    self.kb_danger.add(unknown_cell)
                else:
                    self.kb_safe.add(unknown_cell)

        return safe_moves  
    
    def get_safe_unvisited(self, pos, risk_threshold=0.5):
        import random
        neighbors = self.get_neighbors(pos)

        # Step 1: Safe and Unvisited
        safe_candidates = [
            n for n in neighbors
            if n in self.kb_safe and n not in self.kb_visited
        ]
        if safe_candidates:
            print("🟢 Moving to safe unvisited cell.")
            return [random.choice(safe_candidates)]

        # Step 2: Unknown neighbors (not safe, not visited, not danger) - RISK
        unknown_candidates = [
            n for n in neighbors
            if n not in self.kb_visited and n not in self.kb_safe and n not in self.kb_danger
        ]
        if unknown_candidates:
            if random.random() < risk_threshold:
                print("⚠️ Taking a risk on unknown cell.")
                return [random.choice(unknown_candidates)]
            else:
                print("🛑 Risk too high. Skipping unknowns.")

        # Step 3: Backtrack to safe but visited neighbors
        safe_visited = [
            n for n in neighbors
            if n in self.kb_safe and n in self.kb_visited
        ]
        if safe_visited:
            print("🔁 Backtracking to safe visited cell.")
            return [random.choice(safe_visited)]

        # Nothing to move to
        return []


    def get_neighbors(self, pos):
        x, y = pos
        return [(nx, ny) for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
                if 0 <= nx < self.env.grid_size and 0 <= ny < self.env.grid_size]

    def get_safe_unvisited(self, pos, risk_threshold=0.5):
        import random
        neighbors = self.get_neighbors(pos)

        # Step 1: Prefer definitely safe and unvisited neighbors
        safe_candidates = [
            n for n in neighbors
            if n in self.kb_safe and n not in self.kb_visited
        ]
        if safe_candidates:
            return [random.choice(safe_candidates)]

        # Step 2: If no safe move, consider unknowns (not visited and not marked dangerous)
        unknown_candidates = [
            n for n in neighbors
            if n not in self.kb_visited and n not in self.kb_danger and n not in self.kb_safe
        ]
        if unknown_candidates:
            # Decide whether to take a risk based on probability
            if random.random() < risk_threshold:
                print(f"⚠️ No guaranteed safe moves. Taking a risk!")
                return [random.choice(unknown_candidates)]
            else:
                print(f"🛑 Risk too high. Skipping move.")
                return []

        # Step 3: Nothing left
        return []



    def move_to(self, target):
        self.env.agent_pos = target
        self.env.percepts = self.env.get_percepts()
        self.kb_visited.add(target)
        self.path.append(target)

    def backtrack(self):
        print("🔁 Backtracking...")
        path_to_start = self.find_path_to_target((0, 0), self.env.agent_pos)
        for step in path_to_start:
            # Check if the step is within bounds
            x, y = step
            if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                self.move_to(step)
                print(f"🔙 Backtracked to {step}")
            else:
                print(f"⛔ Invalid backtracking step {step}. Out of bounds.")
                break
    def find_path_to_target(self, target, start):
        from collections import deque

        queue = deque([(start, [])])
        visited = set([start])

        while queue:
            current, path = queue.popleft()

            if current == target:
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor in self.kb_safe:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []
    
# ----------------------
# 3. Visualization Code
# ----------------------
import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
GRID_SIZE = 4
CELL_SIZE = 120
PADDING = 60
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 2 * PADDING
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 2 * PADDING
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Wumpus World - Pygame Visualization")
font = pygame.font.SysFont(None, 24)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SAFE_GREEN = (144, 238, 144)
DANGER_YELLOW = (255, 255, 102)
AGENT_BLUE = (30, 144, 255)

def draw_cell(x, y, color, label=""):
    rect = pygame.Rect(PADDING + y * CELL_SIZE, PADDING + (GRID_SIZE - 1 - x) * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, BLACK, rect, 2)
    if label:
        txt = font.render(label, True, BLACK)
        screen.blit(txt, txt.get_rect(center=rect.center))

def draw_agent(x, y, direction):
    center_x = PADDING + y * CELL_SIZE + CELL_SIZE // 2
    center_y = PADDING + (GRID_SIZE - 1 - x) * CELL_SIZE + CELL_SIZE // 2
    arrow_size = 10

    if direction == "up":
        points = [
            (center_x, center_y - arrow_size),
            (center_x - arrow_size, center_y + arrow_size),
            (center_x + arrow_size, center_y + arrow_size)
        ]
    elif direction == "down":
        points = [
            (center_x, center_y + arrow_size),
            (center_x - arrow_size, center_y - arrow_size),
            (center_x + arrow_size, center_y - arrow_size)
        ]
    elif direction == "left":
        points = [
            (center_x - arrow_size, center_y),
            (center_x + arrow_size, center_y - arrow_size),
            (center_x + arrow_size, center_y + arrow_size)
        ]
    elif direction == "right":
        points = [
            (center_x + arrow_size, center_y),
            (center_x - arrow_size, center_y - arrow_size),
            (center_x - arrow_size, center_y + arrow_size)
        ]

    pygame.draw.polygon(screen, AGENT_BLUE, points)


def visualize_pygame(env, agent):
    screen.fill(WHITE)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            cell = (x, y)
            color = WHITE
            if cell in agent.kb_safe:
                color = SAFE_GREEN
            elif cell in agent.kb_danger:
                color = DANGER_YELLOW

            label = ""
            if env.world[x][y]["pit"]: label += "P"
            if env.world[x][y]["wumpus"] and env.wumpus_alive: label += "W"
            if env.world[x][y]["gold"]: label += "G"
            draw_cell(x, y, color, label)

    ax, ay = env.agent_pos
    draw_cell(ax, ay, WHITE)  # Draw white background to clear text
    draw_agent(ax, ay, env.agent_dir)

    pygame.display.update()


# -----------------
# 4. Main Execution
# -----------------
class KnowledgeBase:
    def __init__(self):
        self.learned_pits = set()
        self.learned_safe = set()
        self.learned_visited = set()

    def update_from_agent(self, agent):
        self.learned_pits.update(agent.kb_danger)
        self.learned_safe.update(agent.kb_safe)
        self.learned_visited.update(agent.kb_visited)

    def apply_to_agent(self, agent):
        agent.kb_danger = self.learned_pits.copy()
        agent.kb_safe = self.learned_safe.copy()
        agent.kb_visited = {(0, 0)}  # Always reset visited to start
        
class KnowledgeBase:
    def __init__(self):
        self.learned_pits = set()
        self.learned_safe = set()
        self.learned_visited = set()

    def update_from_agent(self, agent):
        self.learned_pits.update(agent.kb_danger)
        self.learned_safe.update(agent.kb_safe)
        self.learned_visited.update(agent.kb_visited)

    def apply_to_agent(self, agent):
        agent.kb_danger = self.learned_pits.copy()
        agent.kb_safe = self.learned_safe.copy()
        agent.kb_visited = {(0, 0)}  # Always reset visited to start
        
if __name__ == "__main__":
    # STEP 1: Generate a world once and save it
    temp_env = WumpusWorld()
    saved_world = temp_env.world
    print("🔒 Fixed world generated for a single run.")

    # STEP 2: Knowledge base for cumulative learning
    shared_knowledge = KnowledgeBase()

    print(f"\n🧠=== Starting Run ===")
    # Reuse same world layout
    env = WumpusWorld(world_template=saved_world)
    agent = LogicAgent(env, knowledge=shared_knowledge)

    fig, ax = plt.subplots()
    won = False
    
    while True:
        percepts = env.percepts
        visualize_pygame(env, agent)
        pygame.time.delay(600)

        if percepts["stench"] and env.has_arrow:
            env.shoot_arrow()
            env.percepts = env.get_percepts()

        if percepts["glitter"]:
            env.grab_gold()
            print("Gold grabbed!")
            agent.backtrack()
            won = True
            break

        agent.update_kb(env.agent_pos, percepts)
        next_cells = agent.get_safe_unvisited(env.agent_pos)

        if next_cells:
            agent.move_to(next_cells[0])
        else:
            print("No safe moves. Halting.")
            break

        if env.is_game_over() == "lose":
            print("Agent died.")
            break
    

    shared_knowledge.update_from_agent(agent)
    print("Learned Pits:", shared_knowledge.learned_pits)
    print("Safe Cells:", shared_knowledge.learned_safe)

    if won:
        print("🎯 Run ended with WIN")
    else:
        print("💥 Run ended with FAILURE")

    # Keep the final plot visible
    plt.show()
