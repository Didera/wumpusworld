import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import numpy as np
import time

class WumpusWorld:
    def __init__(self):
        self.grid_size = 4
        self.agent_pos = (0, 0)  # Starting at (1,1) in grid notation
        self.agent_dir = "right"  # Initial direction
        self.has_gold = False
        self.has_arrow = True
        self.wumpus_alive = True
        self.world = self.generate_world()
        self.percepts = self.get_percepts()

    def generate_world(self):
        # Initialize empty grid
        world = [[{"pit": False, "wumpus": False, "gold": False} for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place pits (20% chance per cell, except start)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) != (0, 0) and random.random() < 0.2:
                    world[i][j]["pit"] = True
        
        # Place Wumpus (random, not at start)
        wumpus_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        while wumpus_pos == (0, 0):
            wumpus_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        world[wumpus_pos[0]][wumpus_pos[1]]["wumpus"] = True
        
        # Place gold (random, not at start)
        gold_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        while gold_pos == (0, 0):
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
    def __init__(self, world_size=4):
        self.world_size = world_size
        self.visited = set()
        self.safe_cells = set()
        self.danger_cells = set()
        self.pit_cells = set()
        self.wumpus_cells = set()
        self.knowledge_base = []
        self.path_memory = []
        self.gold_found = False
        
        # Mark starting position as safe
        self.safe_cells.add((0, 0))
    
    def get_adjacent_cells(self, pos):
        """Get adjacent cells within grid bounds"""
        x, y = pos
        adjacent = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.world_size and 0 <= ny < self.world_size:
                adjacent.append((nx, ny))
        return adjacent
    
    def update_knowledge(self, pos, percepts):
        """Update knowledge base with new percepts"""
        self.visited.add(pos)
        self.safe_cells.add(pos)
        
        adjacent_cells = self.get_adjacent_cells(pos)
        
        # If no breeze, adjacent cells are safe from pits
        if not percepts["breeze"]:
            for cell in adjacent_cells:
                self.safe_cells.add(cell)
        else:
            # There's at least one pit in adjacent cells
            unknown_adjacent = [cell for cell in adjacent_cells 
                             if cell not in self.safe_cells and cell not in self.pit_cells]
            if len(unknown_adjacent) == 1:
                self.pit_cells.add(unknown_adjacent[0])
                self.danger_cells.add(unknown_adjacent[0])
        
        # If no stench, adjacent cells are safe from Wumpus
        if not percepts["stench"]:
            for cell in adjacent_cells:
                if cell not in self.wumpus_cells:
                    self.safe_cells.add(cell)
        else:
            # There's Wumpus in one of adjacent cells
            unknown_adjacent = [cell for cell in adjacent_cells 
                             if cell not in self.safe_cells and cell not in self.wumpus_cells]
            if len(unknown_adjacent) == 1:
                self.wumpus_cells.add(unknown_adjacent[0])
                self.danger_cells.add(unknown_adjacent[0])
    
    def get_safe_unvisited_neighbors(self, pos):
        """Get safe unvisited adjacent cells"""
        adjacent = self.get_adjacent_cells(pos)
        safe_unvisited = []
        for cell in adjacent:
            if (cell in self.safe_cells and 
                cell not in self.visited and 
                cell not in self.danger_cells):
                safe_unvisited.append(cell)
        return safe_unvisited
    
    def find_path_to_target(self, start, target):
        """Find shortest path to target using BFS"""
        if start == target:
            return []
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self.get_adjacent_cells(current):
                if neighbor == target:
                    return path + [neighbor]
                
                if (neighbor not in visited and 
                    neighbor in self.safe_cells and 
                    neighbor not in self.danger_cells):
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No safe path found
    
    def get_direction_to_move(self, current_pos, target_pos):
        """Get direction from current position to target"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if dx == -1:
            return "up"
        elif dx == 1:
            return "down"
        elif dy == -1:
            return "left"
        elif dy == 1:
            return "right"
        return None
    
    def should_shoot_arrow(self, env):
        """Determine if agent should shoot arrow"""
        if not env.has_arrow or not env.percepts["stench"]:
            return False, None
        
        # Check if Wumpus is in line of sight
        directions = ["up", "down", "left", "right"]
        x, y = env.agent_pos
        
        for direction in directions:
            if direction == "up":
                positions = [(i, y) for i in range(x-1, -1, -1)]
            elif direction == "down":
                positions = [(i, y) for i in range(x+1, self.world_size)]
            elif direction == "left":
                positions = [(x, j) for j in range(y-1, -1, -1)]
            elif direction == "right":
                positions = [(x, j) for j in range(y+1, self.world_size)]
            
            for pos in positions:
                if pos in self.wumpus_cells:
                    return True, direction
        
        return False, None
    
    def choose_action(self, env):
        """Choose next action based on current state and knowledge"""
        current_pos = env.agent_pos
        percepts = env.percepts
        
        # Update knowledge with current percepts
        self.update_knowledge(current_pos, percepts)
        
        # If gold is present, grab it
        if percepts["glitter"] and not env.has_gold:
            self.gold_found = True
            return "grab_gold"
        
        # If has gold, return to start
        if env.has_gold:
            path = self.find_path_to_target(current_pos, (0, 0))
            if path:
                next_pos = path[0]
                target_dir = self.get_direction_to_move(current_pos, next_pos)
                if env.agent_dir != target_dir:
                    if self.get_turn_direction(env.agent_dir, target_dir) == "left":
                        return "turn_left"
                    else:
                        return "turn_right"
                return "move_forward"
        
        # Check if should shoot arrow
        should_shoot, shoot_dir = self.should_shoot_arrow(env)
        if should_shoot:
            if env.agent_dir != shoot_dir:
                if self.get_turn_direction(env.agent_dir, shoot_dir) == "left":
                    return "turn_left"
                else:
                    return "turn_right"
            return "shoot_arrow"
        
        # Explore safe unvisited cells
        safe_neighbors = self.get_safe_unvisited_neighbors(current_pos)
        if safe_neighbors:
            next_pos = safe_neighbors[0]  # Choose first safe neighbor
            target_dir = self.get_direction_to_move(current_pos, next_pos)
            if env.agent_dir != target_dir:
                if self.get_turn_direction(env.agent_dir, target_dir) == "left":
                    return "turn_left"
                else:
                    return "turn_right"
            return "move_forward"
        
        # Find path to any safe unvisited cell
        for cell in self.safe_cells:
            if cell not in self.visited and cell not in self.danger_cells:
                path = self.find_path_to_target(current_pos, cell)
                if path:
                    next_pos = path[0]
                    target_dir = self.get_direction_to_move(current_pos, next_pos)
                    if env.agent_dir != target_dir:
                        if self.get_turn_direction(env.agent_dir, target_dir) == "left":
                            return "turn_left"
                        else:
                            return "turn_right"
                    return "move_forward"
        
        # If no safe moves, try to go back to start
        if current_pos != (0, 0):
            path = self.find_path_to_target(current_pos, (0, 0))
            if path:
                next_pos = path[0]
                target_dir = self.get_direction_to_move(current_pos, next_pos)
                if env.agent_dir != target_dir:
                    if self.get_turn_direction(env.agent_dir, target_dir) == "left":
                        return "turn_left"
                    else:
                        return "turn_right"
                return "move_forward"
        
        # Default action if stuck
        return "turn_right"
    
    def get_turn_direction(self, current_dir, target_dir):
        """Determine whether to turn left or right to face target direction"""
        dirs = ["up", "right", "down", "left"]
        current_idx = dirs.index(current_dir)
        target_idx = dirs.index(target_dir)
        
        # Calculate the shortest turn
        right_turns = (target_idx - current_idx) % 4
        left_turns = (current_idx - target_idx) % 4
        
        return "right" if right_turns <= left_turns else "left"


class WumpusWorldVisualizer:
    def __init__(self, world, agent):
        self.world = world
        self.agent = agent
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
    def visualize(self, step_count=0):
        """Visualize the current state of the Wumpus World"""
        self.ax.clear()
        self.ax.set_xlim(-0.5, 3.5)
        self.ax.set_ylim(-0.5, 3.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Invert y-axis to match grid notation
        
        # Draw grid
        for i in range(5):
            self.ax.axhline(i-0.5, color='black', linewidth=1)
            self.ax.axvline(i-0.5, color='black', linewidth=1)
        
        # Draw world contents
        for i in range(4):
            for j in range(4):
                cell = self.world.world[i][j]
                
                # Color cells based on agent's knowledge
                if (i, j) in self.agent.visited:
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='lightgreen', alpha=0.3)
                    self.ax.add_patch(rect)
                elif (i, j) in self.agent.safe_cells:
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='lightblue', alpha=0.3)
                    self.ax.add_patch(rect)
                elif (i, j) in self.agent.danger_cells:
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='red', alpha=0.3)
                    self.ax.add_patch(rect)
                
                # Draw pits
                if cell["pit"]:
                    circle = patches.Circle((j, i), 0.3, color='black', alpha=0.8)
                    self.ax.add_patch(circle)
                    self.ax.text(j, i, 'P', ha='center', va='center', 
                               color='white', fontsize=12, fontweight='bold')
                
                # Draw Wumpus
                if cell["wumpus"] and self.world.wumpus_alive:
                    triangle = patches.RegularPolygon((j, i), 3, 0.3, 
                                                    color='red', alpha=0.8)
                    self.ax.add_patch(triangle)
                    self.ax.text(j, i, 'W', ha='center', va='center', 
                               color='white', fontsize=12, fontweight='bold')
                
                # Draw gold
                if cell["gold"]:
                    star = patches.RegularPolygon((j, i), 5, 0.2, 
                                                color='gold', alpha=0.8)
                    self.ax.add_patch(star)
                    self.ax.text(j, i, 'G', ha='center', va='center', 
                               color='black', fontsize=10, fontweight='bold')
        
        # Draw agent
        agent_x, agent_y = self.world.agent_pos
        agent_color = 'blue'
        if self.world.has_gold:
            agent_color = 'orange'
        
        agent_circle = patches.Circle((agent_y, agent_x), 0.25, 
                                    color=agent_color, alpha=0.9)
        self.ax.add_patch(agent_circle)
        
        # Draw agent direction arrow
        directions = {"up": (0, -0.3), "down": (0, 0.3), 
                     "left": (-0.3, 0), "right": (0.3, 0)}
        dx, dy = directions[self.world.agent_dir]
        self.ax.arrow(agent_y, agent_x, dx, dy, head_width=0.1, 
                     head_length=0.1, fc='white', ec='white')
        
        # Add percepts information
        percepts_text = f"Step: {step_count}\n"
        percepts_text += f"Position: {self.world.agent_pos}\n"
        percepts_text += f"Direction: {self.world.agent_dir}\n"
        percepts_text += f"Percepts: {self.world.percepts}\n"
        percepts_text += f"Has Gold: {self.world.has_gold}\n"
        percepts_text += f"Has Arrow: {self.world.has_arrow}"
        
        self.ax.text(4.5, 0, percepts_text, fontsize=10, verticalalignment='top')
        
        # Legend
        legend_text = "Legend:\n"
        legend_text += "🔵 Agent\n🟠 Agent with Gold\n"
        legend_text += "⚫ Pit\n🔺 Wumpus\n⭐ Gold\n"
        legend_text += "🟢 Visited\n🔵 Safe\n🔴 Danger"
        
        self.ax.text(4.5, 2, legend_text, fontsize=9, verticalalignment='top')
        
        self.ax.set_title(f'Wumpus World - Logic-Based Agent (Step {step_count})')
        plt.pause(0.5)


def run_agent_simulation(visualize=True, max_steps=100):
    """Run the logic-based agent in the Wumpus World"""
    print("=== Wumpus World Logic-Based Agent Simulation ===\n")
    
    # Create world and agent
    world = WumpusWorld()
    agent = LogicBasedAgent()
    
    if visualize:
        visualizer = WumpusWorldVisualizer(world, agent)
        plt.ion()  # Turn on interactive mode
    
    step_count = 0
    game_result = "continue"
    
    print(f"Initial state:")
    print(f"Agent position: {world.agent_pos}")
    print(f"Agent direction: {world.agent_dir}")
    print(f"Initial percepts: {world.percepts}")
    print("-" * 50)
    
    while game_result == "continue" and step_count < max_steps:
        step_count += 1
        
        # Get agent's action
        action = agent.choose_action(world)
        
        print(f"Step {step_count}: Action = {action}")
        print(f"Current position: {world.agent_pos}, Direction: {world.agent_dir}")
        
        # Execute action
        if action == "move_forward":
            success = world.move_forward()
            print(f"Move forward: {'Success' if success else 'Blocked (Bump)'}")
        elif action == "turn_left":
            world.turn_left()
            print(f"Turned left, now facing: {world.agent_dir}")
        elif action == "turn_right":
            world.turn_right()
            print(f"Turned right, now facing: {world.agent_dir}")
        elif action == "grab_gold":
            success = world.grab_gold()
            print(f"Grab gold: {'Success' if success else 'No gold here'}")
        elif action == "shoot_arrow":
            success = world.shoot_arrow()
            print(f"Shoot arrow: {'Hit Wumpus!' if success else 'Missed'}")
        
        print(f"New percepts: {world.percepts}")
        print(f"Agent knowledge - Visited: {len(agent.visited)}, Safe: {len(agent.safe_cells)}")
        
        # Check game state
        game_result = world.is_game_over()
        
        # Visualize if enabled
        if visualize:
            visualizer.visualize(step_count)
        
        print("-" * 50)
        
        if game_result != "continue":
            break
    
    # Final results
    print(f"\n=== GAME OVER ===")
    print(f"Result: {game_result.upper()}")
    print(f"Total steps: {step_count}")
    print(f"Final position: {world.agent_pos}")
    print(f"Has gold: {world.has_gold}")
    print(f"Cells explored: {len(agent.visited)}/{world.grid_size**2}")
    
    if visualize:
        plt.ioff()  # Turn off interactive mode
        plt.show()
    
    return game_result, step_count


def run_multiple_tests(num_tests=10):
    """Run multiple test simulations to evaluate agent performance"""
    print(f"=== Running {num_tests} Test Simulations ===\n")
    
    results = {"win": 0, "lose": 0, "continue": 0}
    total_steps = 0
    
    for i in range(num_tests):
        print(f"Test {i+1}/{num_tests}")
        result, steps = run_agent_simulation(visualize=False, max_steps=50)
        results[result] += 1
        total_steps += steps
        print(f"Result: {result}, Steps: {steps}\n")
    
    # Print summary
    print("=== TEST SUMMARY ===")
    print(f"Total tests: {num_tests}")
    print(f"Wins: {results['win']} ({results['win']/num_tests*100:.1f}%)")
    print(f"Losses: {results['lose']} ({results['lose']/num_tests*100:.1f}%)")
    print(f"Timeouts: {results['continue']} ({results['continue']/num_tests*100:.1f}%)")
    print(f"Average steps: {total_steps/num_tests:.1f}")
    
    return results


if __name__ == "__main__":
    print("Wumpus World Logic-Based Agent")
    print("1. Run single simulation with visualization")
    print("2. Run multiple test simulations")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_agent_simulation(visualize=True)
    elif choice == "2":
        num_tests = int(input("Enter number of tests (default 10): ") or 10)
        run_multiple_tests(num_tests)
    else:
        print("Running default single simulation...")
        run_agent_simulation(visualize=True)