import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import time

# Set font for better Unicode support
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

class WumpusWorld:
    def _init_(self):
        self.grid_size = 4
        self.agent_pos = (0, 0)  # Starting at (0,0) in grid notation
        self.agent_dir = "right"  # Initial direction
        self.has_gold = False
        self.has_arrow = True
        self.wumpus_alive = True
        self.world = self.generate_world()
        self.percepts = self.get_percepts()

    def generate_world(self):
        # Initialize empty grid
        world = [[{"pit": False, "wumpus": False, "gold": False} for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place Wumpus (random, not at start)
        wumpus_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        while wumpus_pos == (0, 0):
            wumpus_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        world[wumpus_pos[0]][wumpus_pos[1]]["wumpus"] = True

        # Place pits (20% chance per cell, except start, wumpus cell)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) != (0, 0) and (i, j) != wumpus_pos and random.random() < 0.2:
                    world[i][j]["pit"] = True

        # Place gold (random, not at start, not at wumpus or pit)
        gold_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        while gold_pos == (0, 0) or gold_pos == wumpus_pos or world[gold_pos[0]][gold_pos[1]]["pit"]:
            gold_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        world[gold_pos[0]][gold_pos[1]]["gold"] = True
        
        return world

    def get_percepts(self):
        x, y = self.agent_pos
        cell = self.world[x][y]
        percepts = {
            "stench": False,
            "breeze": False,
            "glitter": False,
            "bump": False,
            "scream": False
        }
        
        # Check adjacent cells for Wumpus (stench) and pits (breeze)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.world[nx][ny]["wumpus"] and self.wumpus_alive:
                    percepts["stench"] = True
                if self.world[nx][ny]["pit"]:
                    percepts["breeze"] = True
        
        # Current cell percepts
        if cell["gold"]:
            percepts["glitter"] = True
            
        return percepts

    def move_forward(self):
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        if self.agent_dir == "up":
            new_x -= 1
        elif self.agent_dir == "down":
            new_x += 1
        elif self.agent_dir == "left":
            new_y -= 1
        elif self.agent_dir == "right":
            new_y += 1
        
        # Check if move is valid
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            self.agent_pos = (new_x, new_y)
            self.percepts = self.get_percepts()
            return True
        else:
            self.percepts["bump"] = True
            return False

    def turn_left(self):
        dirs = ["up", "left", "down", "right"]
        idx = dirs.index(self.agent_dir)
        self.agent_dir = dirs[(idx + 1) % 4]
        self.percepts = self.get_percepts()

    def turn_right(self):
        dirs = ["up", "right", "down", "left"]
        idx = dirs.index(self.agent_dir)
        self.agent_dir = dirs[(idx + 1) % 4]
        self.percepts = self.get_percepts()

    def shoot_arrow(self):
        if not self.has_arrow:
            return False
        
        self.has_arrow = False
        x, y = self.agent_pos
        wumpus_killed = False
        
        # Arrow travels in a straight line in the current direction
        if self.agent_dir == "up":
            for i in range(x - 1, -1, -1):
                if self.world[i][y]["wumpus"]:
                    wumpus_killed = True
                    break
        elif self.agent_dir == "down":
            for i in range(x + 1, self.grid_size):
                if self.world[i][y]["wumpus"]:
                    wumpus_killed = True
                    break
        elif self.agent_dir == "left":
            for j in range(y - 1, -1, -1):
                if self.world[x][j]["wumpus"]:
                    wumpus_killed = True
                    break
        elif self.agent_dir == "right":
            for j in range(y + 1, self.grid_size):
                if self.world[x][j]["wumpus"]:
                    wumpus_killed = True
                    break
        
        if wumpus_killed:
            self.wumpus_alive = False
            self.percepts["scream"] = True
            return True
        
        return False

    def grab_gold(self):
        x, y = self.agent_pos
        if self.world[x][y]["gold"]:
            self.has_gold = True
            self.world[x][y]["gold"] = False
            self.percepts["glitter"] = False
            return True
        return False

    def is_game_over(self):
        x, y = self.agent_pos
        cell = self.world[x][y]
        
        if (cell["pit"] or (cell["wumpus"] and self.wumpus_alive)):
            return "lose"
        
        if self.has_gold and self.agent_pos == (0, 0):
            return "win"
        
        return "continue"

class LogicBasedAgent:
    def _init_(self, world_size=4):
        self.world_size = world_size
        self.visited = set()
        self.safe_cells = set()
        self.dangerous_cells = set()
        self.pit_possible = set()
        self.wumpus_possible = set()
        self.knowledge_base = []
        self.path_to_gold = []
        self.gold_location = None
        self.returning_home = False
        self.move_history = []
        
    def add_knowledge(self, cell, percepts):
        """Add knowledge to the knowledge base based on percepts"""
        x, y = cell
        self.visited.add(cell)
        self.safe_cells.add(cell)
        
        # Remove this cell from dangerous possibilities
        self.pit_possible.discard(cell)
        self.wumpus_possible.discard(cell)
        
        neighbors = self.get_neighbors(cell)
        
        if not percepts["breeze"] and not percepts["stench"]:
            # No dangers nearby - all neighbors are safe
            for neighbor in neighbors:
                self.safe_cells.add(neighbor)
                self.pit_possible.discard(neighbor)
                self.wumpus_possible.discard(neighbor)
        else:
            if percepts["breeze"]:
                # There's a pit nearby
                for neighbor in neighbors:
                    if neighbor not in self.visited:
                        self.pit_possible.add(neighbor)
            
            if percepts["stench"]:
                # Wumpus is nearby
                for neighbor in neighbors:
                    if neighbor not in self.visited:
                        self.wumpus_possible.add(neighbor)
    
    def get_neighbors(self, cell):
        """Get valid neighboring cells"""
        x, y = cell
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.world_size and 0 <= ny < self.world_size:
                neighbors.append((nx, ny))
        return neighbors
    
    def deduce_safe_moves(self):
        """Use logical deduction to find safe moves"""
        safe_moves = []
        
        # Find cells that are definitely safe
        for cell in self.safe_cells:
            if cell not in self.visited:
                safe_moves.append(cell)
        
        # Advanced deduction: if we have constraints, try to deduce more
        self.advanced_deduction()
        
        return safe_moves
    
    def advanced_deduction(self):
        """More sophisticated logical deduction"""
        # If a cell is adjacent to a breeze/stench and all other neighbors 
        # are known safe, then this cell must contain the danger
        
        for cell in self.visited:
            neighbors = self.get_neighbors(cell)
            unknown_neighbors = [n for n in neighbors if n not in self.visited]
            
            if len(unknown_neighbors) == 1:
                unknown_cell = unknown_neighbors[0]
                
                # Check if this unknown cell must be dangerous
                pit_count = sum(1 for n in neighbors if n in self.pit_possible)
                wumpus_count = sum(1 for n in neighbors if n in self.wumpus_possible)
                
                if pit_count == 1 and unknown_cell in self.pit_possible:
                    # This cell definitely has a pit
                    self.dangerous_cells.add(unknown_cell)
                    self.pit_possible.discard(unknown_cell)
                
                if wumpus_count == 1 and unknown_cell in self.wumpus_possible:
                    # This cell definitely has the wumpus
                    self.dangerous_cells.add(unknown_cell)
                    self.wumpus_possible.discard(unknown_cell)
    
    def get_next_action(self, env):
        """Determine the next action based on current state and knowledge"""
        current_pos = env.agent_pos
        percepts = env.percepts
        
        # Add current knowledge
        self.add_knowledge(current_pos, percepts)
        
        # Check if gold is here
        if percepts["glitter"] and not env.has_gold:
            self.gold_location = current_pos
            return "grab_gold"
        
        # If we have gold and we're at start, we win!
        if env.has_gold and current_pos == (0, 0):
            return "exit"  # Game should end
        
        # If we have gold, return to start
        if env.has_gold and not self.returning_home:
            self.returning_home = True
            self.path_to_gold = self.find_path_to_target((0, 0), current_pos)
        
        # If we're returning home, follow the path
        if self.returning_home and self.path_to_gold:
            next_cell = self.path_to_gold.pop(0)
            return self.get_move_to_cell(current_pos, next_cell, env.agent_dir)
        
        # Find safe moves
        safe_moves = self.deduce_safe_moves()
        
        if safe_moves:
            # Choose the closest safe move
            target = min(safe_moves, key=lambda x: abs(x[0] - current_pos[0]) + abs(x[1] - current_pos[1]))
            return self.get_move_to_cell(current_pos, target, env.agent_dir)
        
        # If no safe moves, try to shoot wumpus if we detect stench
        if percepts["stench"] and env.has_arrow:
            # Find direction to potential wumpus
            wumpus_cells = [cell for cell in self.wumpus_possible if cell in self.get_neighbors(current_pos)]
            if wumpus_cells:
                target = wumpus_cells[0]
                required_dir = self.get_direction_to_cell(current_pos, target)
                if env.agent_dir != required_dir:
                    return self.get_turn_action(env.agent_dir, required_dir)
                else:
                    return "shoot_arrow"
        
        # Last resort: explore randomly among unvisited cells
        unvisited = []
        for i in range(self.world_size):
            for j in range(self.world_size):
                if (i, j) not in self.visited and (i, j) not in self.dangerous_cells:
                    unvisited.append((i, j))
        
        if unvisited:
            target = min(unvisited, key=lambda x: abs(x[0] - current_pos[0]) + abs(x[1] - current_pos[1]))
            return self.get_move_to_cell(current_pos, target, env.agent_dir)
        
        return "wait"  # No valid moves
    
    def find_path_to_target(self, target, start):
        """Find path from start to target using BFS"""
        queue = deque([(start, [])])
        visited = set([start])
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor in self.safe_cells:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def get_move_to_cell(self, current_pos, target_pos, current_dir):
        """Get the action needed to move toward target cell"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if dx == 0 and dy == 0:
            return "wait"
        
        # Determine required direction
        if abs(dx) > abs(dy):
            required_dir = "down" if dx > 0 else "up"
        else:
            required_dir = "right" if dy > 0 else "left"
        
        if current_dir == required_dir:
            return "move_forward"
        else:
            return self.get_turn_action(current_dir, required_dir)
    
    def get_direction_to_cell(self, current_pos, target_pos):
        """Get the direction from current position to target"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if abs(dx) > abs(dy):
            return "down" if dx > 0 else "up"
        else:
            return "right" if dy > 0 else "left"
    
    def get_turn_action(self, current_dir, required_dir):
        """Get the turn action needed to face the required direction"""
        dirs = ["up", "right", "down", "left"]
        current_idx = dirs.index(current_dir)
        required_idx = dirs.index(required_dir)
        
        diff = (required_idx - current_idx) % 4
        
        if diff == 1:
            return "turn_right"
        elif diff == 3:
            return "turn_left"
        elif diff == 2:
            return "turn_right"  # Two rights = 180 turn
        else:
            return "wait"

import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import time

# Set font for better Unicode support
plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji', 'Arial Unicode MS', 'sans-serif']

class WumpusWorldVisualizer:
    def _init_(self, env, agent):
        self.env = env
        self.agent = agent
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.history = []
        self.fig.subplots_adjust(top=0.82)  # Add space at the top for the status bar

    def animate_arrow(self, direction):
        """Animate the arrow flying in the given direction"""
        arrow_emoji = {'up': '🡅', 'down': '🡇', 'left': '🡄', 'right': '🡆'}
        x, y = self.env.agent_pos
        dx, dy = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}[direction]
        for step in range(1, self.env.grid_size):
            nx, ny = x + dx * step, y + dy * step
            if 0 <= nx < self.env.grid_size and 0 <= ny < self.env.grid_size:
                self.visualize_current_state()
                self.ax.text(ny, self.env.grid_size - 1 - nx, arrow_emoji[direction], ha='center', va='center', fontsize=32, color='orange')
                plt.pause(0.15)
            else:
                break

    def animate_defeat(self, cause):
        """Show animation when agent is defeated"""
        x, y = self.env.agent_pos
        agent_x, agent_y = y, self.env.grid_size - 1 - x
        self.visualize_current_state()
        if cause == "wumpus":
            self.ax.text(agent_x, agent_y, "💀😱", ha='center', va='center', fontsize=40)
        elif cause == "pit":
            self.ax.text(agent_x, agent_y, "🕳😱", ha='center', va='center', fontsize=40)
        plt.pause(1)

    def visualize_current_state(self):
        """Visualize the current state of the world with emojis and percept animations"""
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.env.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.env.grid_size - 0.5)
        self.ax.set_aspect('equal')

        # Draw grid
        for i in range(self.env.grid_size + 1):
            self.ax.axhline(i - 0.5, color='black', linewidth=1)
            self.ax.axvline(i - 0.5, color='black', linewidth=1)

        # Draw world contents with emojis and backgrounds
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                cell = self.env.world[i][j]
                cell_x = j
                cell_y = self.env.grid_size - 1 - i

                # Draw background for pit or wumpus
                if cell["pit"]:
                    self.ax.add_patch(
                        patches.Rectangle((cell_x - 0.5, cell_y - 0.5), 1, 1, color='lightblue', alpha=0.5, zorder=0)
                    )
                if cell["wumpus"]:
                    self.ax.add_patch(
                        patches.Rectangle((cell_x - 0.5, cell_y - 0.5), 1, 1, color='mistyrose', alpha=0.5, zorder=0)
                    )

                # Draw only one emoji per cell, by priority and context
                if (i, j) == self.env.agent_pos:
                    # Agent on pit (lose)
                    if cell["pit"]:
                        self.ax.text(cell_x, cell_y, "😮", ha='center', va='center', fontsize=36, fontweight='bold')
                    # Agent on wumpus (lose)
                    elif cell["wumpus"] and self.env.wumpus_alive:
                        self.ax.text(cell_x, cell_y, "😫", ha='center', va='center', fontsize=36, fontweight='bold')
                    # Agent on gold (win)
                    elif cell["gold"] or self.env.has_gold:
                        self.ax.text(cell_x, cell_y, "😄", ha='center', va='center', fontsize=36, fontweight='bold')
                    else:
                        self.ax.text(cell_x, cell_y, "👀", ha='center', va='center', fontsize=36, fontweight='bold')
                elif cell["wumpus"]:
                    emoji = "💀" if self.env.wumpus_alive else "☠"
                    self.ax.text(cell_x, cell_y, emoji, ha='center', va='center', fontsize=36, fontweight='bold')
                elif cell["pit"]:
                    self.ax.text(cell_x, cell_y, "🕳", ha='center', va='center', fontsize=36, fontweight='bold')
                elif cell["gold"]:
                    self.ax.text(cell_x, cell_y, "🪙✨", ha='center', va='center', fontsize=32, fontweight='bold')

        # Animate percepts: breeze (💨) and stench (💨 in green)
        # Show breeze and stench ONLY in the agent's current cell if perceived
        x, y = self.env.agent_pos
        cell_x, cell_y = y, self.env.grid_size - 1 - x
        if self.env.percepts.get("breeze", False):
            self.ax.text(cell_x, cell_y + 0.35, "breeze 💨", ha='center', va='center', fontsize=16, color='darkblue')
        if self.env.percepts.get("stench", False):
            self.ax.text(cell_x, cell_y + 0.35, "stench 💨", ha='center', va='center', fontsize=16, color='green')
        # Draw agent
        agent_x, agent_y = y, self.env.grid_size - 1 - x
        self.ax.text(agent_x, agent_y, "", ha='center', va='center', fontsize=36, fontweight='bold')
        # Agent direction arrow (emoji)
        dir_emoji = {"up": "⬆", "down": "⬇", "left": "⬅", "right": "➡"}
        dx, dy = {"up": (0, 0.3), "down": (0, -0.3), "left": (-0.3, 0), "right": (0.3, 0)}[self.env.agent_dir]
        self.ax.text(agent_x + dx, agent_y + dy, dir_emoji[self.env.agent_dir], ha='center', va='center', fontsize=18)

        # # Add status text
        # status = f"Position: {self.env.agent_pos}, Direction: {self.env.agent_dir}\n"
        # status += f"Percepts: {self.env.percepts}\n"
        # status += f"Has Gold: {self.env.has_gold}, Has Arrow: {self.env.has_arrow}\n"
        # status += f"Visited: {len(self.agent.visited)} cells"
        # # Remove any previous status bar
        # if hasattr(self, 'status_bar'):
        #     self.status_bar.remove()
        # # Place status bar above the axes
        # self.status_bar = self.fig.text(
        #     0.01, 0.93, status, fontsize=11, va='top', ha='left',
        #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        # )

        # Legend with emojis
        # legend_elements = [
        #     plt.Line2D([0], [0], marker='$😃$', color='w', label='Agent', markersize=18, linestyle='None'),
        #     plt.Line2D([0], [0], marker='$🕳$', color='w', label='Pit', markersize=18, linestyle='None'),
        #     plt.Line2D([0], [0], marker='$💀$', color='w', label='Wumpus', markersize=18, linestyle='None'),
        #     plt.Line2D([0], [0], marker='$🪙$', color='w', label='Gold', markersize=18, linestyle='None'),
        #     plt.Line2D([0], [0], marker='$💨$', color='w', label='Breeze', markersize=18, linestyle='None'),
        #     plt.Line2D([0], [0], marker='$💨$', color='green', label='Stench', markersize=18, linestyle='None'),
        # ]
        # self.ax.legend(handles=legend_elements, loc='upper right')

        self.ax.set_title('Wumpus World - Logic-Based Agent', fontsize=14, fontweight='bold')
        plt.pause(0.1)

        # Save current state to history
        self.history.append({
            'agent_pos': self.env.agent_pos,
            'agent_dir': self.env.agent_dir,
            'percepts': self.env.percepts.copy(),
            'has_gold': self.env.has_gold,
            'visited': self.agent.visited.copy()
        })
def run_simulation():
    """Run a complete simulation of the Wumpus World"""
    print("=== Wumpus World Logic-Based Agent Simulation ===\n")
    
    # Create environment and agent
    env = WumpusWorld()
    agent = LogicBasedAgent()
    visualizer = WumpusWorldVisualizer(env, agent)
    
    # Print initial world state (for debugging)
    print("World Layout:")
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            cell = env.world[i][j]
            contents = []
            if cell["pit"]: contents.append("P")
            if cell["wumpus"]: contents.append("W")
            if cell["gold"]: contents.append("G")
            if not contents: contents.append(".")
            print(f"{''.join(contents):>3}", end=" ")
        print()
    print()
    
    step = 0
    max_steps = 100
    
    plt.ion()  # Interactive mode for real-time visualization
    
    while step < max_steps:
        print(f"=== Step {step + 1} ===")
        print(f"Agent at {env.agent_pos}, facing {env.agent_dir}")
        print(f"Percepts: {env.percepts}")
        
        # Visualize current state
        visualizer.visualize_current_state()
        
        # Check game status
        game_status = env.is_game_over()
        if game_status != "continue":
            x, y = env.agent_pos
            cell = env.world[x][y]
            if game_status == "lose":
                if cell["wumpus"]:
                    visualizer.visualize_current_state()
                    plt.pause(2)
                    print("\n=== GAME OVER: LOSE (Wumpus) ===")
                    break
                elif cell["pit"]:
                    visualizer.visualize_current_state()
                    plt.pause(2)
                    print("\n=== GAME OVER: LOSE (Pit) ===")
                    break
            elif game_status == "win":
                if cell["gold"]:
                    visualizer.visualize_current_state()
                    plt.pause(2)
                    print("\n=== GAME OVER: WIN (Gold) ===")
                    break
        
        # Get next action from agent
        action = agent.get_next_action(env)
        print(f"Agent decides to: {action}")
        
        # Execute action
        if action == "move_forward":
            success = env.move_forward()
            if not success:
                print("Hit a wall!")
        elif action == "turn_left":
            env.turn_left()
        elif action == "turn_right":
            env.turn_right()
        elif action == "grab_gold":
            success = env.grab_gold()
            if success:
                print("Gold collected!")
                visualizer.visualize_current_state()
                plt.pause(2)
                print("\n=== GAME OVER: WIN (Gold) ===")
                break
        elif action == "shoot_arrow":
            direction = env.agent_dir
            visualizer.animate_arrow(direction)
            success = env.shoot_arrow()
            if success:
                print("Wumpus killed!")
            else:
                print("Arrow missed!")
        elif action == "wait":
            print("Agent is waiting...")
            time.sleep(1)
        elif action == "exit":
            print("Agent exits the world!")
            break
        
        print()
        step += 1
        time.sleep(0.5)  # Pause for visualization
    
    plt.ioff()  # Turn off interactive mode
    
    # Final statistics
    print(f"\nSimulation completed in {step} steps")
    print(f"Agent visited {len(agent.visited)} cells")
    print(f"Final status: {env.is_game_over()}")
    
    # Keep the final visualization open
    plt.show()
    
    return env, agent, visualizer

def test_multiple_runs(num_runs=5):
    """Test the agent across multiple random worlds"""
    print(f"=== Testing Agent Across {num_runs} Random Worlds ===\n")
    
    wins = 0
    total_steps = 0
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        # Set random seed for reproducible testing
        random.seed(run * 42)
        
        env = WumpusWorld()
        agent = LogicBasedAgent()
        
        step = 0
        max_steps = 100
        
        while step < max_steps:
            game_status = env.is_game_over()
            if game_status != "continue":
                if game_status == "win":
                    wins += 1
                    print(f"  WIN in {step} steps!")
                else:
                    print(f"  LOSS in {step} steps")
                break
            
            action = agent.get_next_action(env)
            
            # Execute action
            if action == "move_forward":
                env.move_forward()
            elif action == "turn_left":
                env.turn_left()
            elif action == "turn_right":
                env.turn_right()
            elif action == "grab_gold":
                env.grab_gold()
            elif action == "shoot_arrow":
                env.shoot_arrow()
            elif action == "exit":
                break
            
            step += 1
        
        total_steps += step
        print()
    
    print(f"=== Test Results ===")
    print(f"Wins: {wins}/{num_runs} ({wins/num_runs*100:.1f}%)")
    print(f"Average steps per game: {total_steps/num_runs:.1f}")

if __name__ == "_main_":
    # Run single simulation with visualization
    print("Running single simulation with visualization...")
    env, agent, viz = run_simulation()
    
    # Run multiple test runs
    print("\n" + "="*50)
    test_multiple_runs(5)
    
    print("\n=== Assignment Completion ===")
    print("✓ 1. Logic-based agent implemented")
    print("✓ 2. Visualization added with matplotlib")
    print("✓ 3. Agent tested and demonstrated")
    print("✓ 4. Report ready (see comments and output)")