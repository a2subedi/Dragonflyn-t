import numpy as np
import random
import copy

class Environment:
    def __init__(self, size, num_obstacles, num_food, num_agents):
        self.size = size
        self.grid = np.full((size, size), ' ', dtype=str)  # Empty cells as ' '
        self.num_obstacles = num_obstacles
        self.num_food = num_food
        self.num_agents = num_agents
        self.agents : list[Agent]= []
        
        self._place_obstacles()
        self._place_food()
        self._place_agents()

    def _place_obstacles(self):
        for _ in range(self.num_obstacles):
            while True:
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                if self.grid[x, y] == ' ':
                    self.grid[x, y] = 'X'  # Obstacles as 'X'
                    break

    def _place_food(self):
        for _ in range(self.num_food):
            while True:
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                if self.grid[x, y] == ' ':
                    self.grid[x, y] = 'F'  # Food as 'F'
                    break

    def _place_agents(self):
        for i in range(self.num_agents):
            while True:
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                if self.grid[x, y] == ' ':
                    agent = Agent(x, y, self.size)
                    self.agents.append(agent)
                    break


    def display(self):
        display_grid = self.grid.copy()
        # single_agent = Agent(5,5,self.size)
        for agent in self.agents:
            # _x,_y = single_agent.get_pos()
            # print("x:",_x, "y:",_y)
            print(agent[0].get_pos())
            # print(agent.y)
            # display_grid[agent.x, agent.y] = 'A'  # Agents as 'A'
            display_grid[agent[0].get_pos()] = 'A'  # Agents as 'A'
        print(display_grid)


class Agent:
    def __init__(self, x, y, grid_size, vision_radius=1):
        self.x = x
        self.y = y
        self.grid_size = grid_size
        self.vision_radius = vision_radius
        self.gene = self._generate_random_gene()
        self.fitness = 0
        self.food_collected = 0

    def _generate_random_gene(self):
        actions = ['up', 'down', 'left', 'right', 'stay']
        gene_length = 10  # The number of rules or conditions
        return [(random.randint(0, 2), random.choice(actions)) for _ in range(gene_length)]

    def perceive(self, environment):
        # Extract the local grid in the vision range
        min_x = max(0, self.x - self.vision_radius)
        max_x = min(self.grid_size - 1, self.x + self.vision_radius)
        min_y = max(0, self.y - self.vision_radius)
        max_y = min(self.grid_size - 1, self.y + self.vision_radius)

        perception = environment.grid[min_x:max_x+1, min_y:max_y+1]
        return perception

    def decide(self, perception):
        # Decision based on perception and gene
        for condition, action in self.gene:
            if condition == 2 and 'F' in perception:  # Prioritize food
                return action
            if condition == 1 and 'X' in perception:  # Avoid obstacles
                return action
        return random.choice(['up', 'down', 'left', 'right'])  # Default random move

    def act(self, action):
        if action == 'up' and self.x > 0:
            self.x -= 1
        elif action == 'down' and self.x < self.grid_size - 1:
            self.x += 1
        elif action == 'left' and self.y > 0:
            self.y -= 1
        elif action == 'right' and self.y < self.grid_size - 1:
            self.y += 1

    def update_fitness(self, environment):
        if environment.grid[self.x, self.y] == 'F':
            self.food_collected += 1
            self.fitness += 10
            environment.grid[self.x, self.y] = ' '  # Remove the food from the grid

    def get_pos(self):
        return self.x,self.y


def simulate_generation(environment, num_steps):
    for _ in range(num_steps):
        for agent in environment.agents:
            perception = agent[0].perceive(environment)
            action = agent.decide(perception)
            agent.act(action)
            agent.update_fitness(environment)


def evaluate_population(environment):
    for agent in environment.agents:
        print(f"Agent Fitness: {agent.fitness}, Food Collected: {agent.food_collected}")


def select_agents(agents, num_to_select):
    agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
    return agents[:num_to_select]


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1.gene) - 1)
    child_gene = parent1.gene[:crossover_point] + parent2.gene[crossover_point:]
    return Agent(random.randint(0, parent1.grid_size - 1), random.randint(0, parent1.grid_size - 1), parent1.grid_size, vision_radius=parent1.vision_radius), child_gene


def mutate(agent, mutation_rate=0.01):
    if random.random() < mutation_rate:
        mutation_point = random.randint(0, len(agent.gene) - 1)
        actions = ['up', 'down', 'left', 'right', 'stay']
        agent.gene[mutation_point] = (random.randint(0, 2), random.choice(actions))


def next_generation(environment, num_to_select, mutation_rate):
    selected_agents = select_agents(environment.agents, num_to_select)
    new_agents = []

    while len(new_agents) < len(environment.agents):
        parent1, parent2 = random.sample(selected_agents, 2)
        child = crossover(parent1, parent2)
        mutate(child, mutation_rate)
        new_agents.append(child)

    environment.agents = new_agents


def main():
    size = 10  # Size of the grid (NxN)
    num_obstacles = 10
    num_food = 5
    num_agents = 10
    num_generations = 20
    num_steps_per_generation = 50

    environment = Environment(size, num_obstacles, num_food, num_agents)

    for generation in range(num_generations):
        print(f"Generation {generation + 1}")
        simulate_generation(environment, num_steps_per_generation)
        evaluate_population(environment)
        next_generation(environment, num_to_select=5, mutation_rate=0.05)
        environment.display()

if __name__ == "__main__":
    main()
