import numpy as np
import random
# import copy


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
        # for i in range(1):
            while True:
                x, y =  random.randint(0, self.size-1), random.randint(0, self.size-1)
                # x, y = self.size-1,self.size-1
                if self.grid[x, y] == ' ':
                    self.grid[x, y] = i  # Food as 'F'
                    agent = Agent(x, y, self.size,1,'G1_'+str(i))
                    self.agents.append(agent)
                    break


    def display(self):
        display_grid = self.grid.copy()
        # for count, agent in enumerate(self.agents):
        #     # display_grid[agent.x, agent.y] = 'A'  # Agents as 'A'
        #     if type(agent) is Agent:
        #         display_grid[agent.get_pos()] =  str(count)  # Agents as 'A'
        #     if type(agent) is tuple:
        #         display_grid[agent[0].get_pos()] =  str(count)  # Agents as 'A'
        print(display_grid,'\n')



class Agent:
    # class variables
    # points = [ [y*(-1),x*(-1)] for x in range(-1,2) for y in range(1,-2,-1) ]
    # vec = np.array(points)
    # print('vec',vec)

    def __init__(self, x, y, grid_size, vision_radius=1, name = None):
        self.x = x
        self.y = y
        self.grid_size = grid_size
        self.vision_radius = vision_radius
        self.gene = self._generate_random_gene()
        self.fitness = 100
        self.food_collected = 0
        self.name = name

    def _generate_random_gene(self):
        '''Generates Sequqnce of genes. 3*3*3*5 array'''
        # actions = ['u', 'd', 'l', 'r', 's']
        # exit()
        # gene_length = 18  # The number of rules or conditions
        # return [( _, random.choice(actions)) for _ in range(gene_length)]
        genes = np.array([random_probabilities() for _ in range(27)]).reshape(3,3,3,5)
        # print(np.size(genes),'genesize')
        return genes
        # print(genes)


    def perceive(self, environment):
        vison_diameter = (self.vision_radius*2) + 1
        # Extract the local grid in the vision range
        min_x = max(0, self.x - self.vision_radius)
        max_x = min(self.grid_size - 1, self.x + self.vision_radius)
        min_y = max(0, self.y - self.vision_radius)
        max_y = min(self.grid_size - 1, self.y + self.vision_radius)

        local_grid = np.full((vison_diameter, vison_diameter), ' ', dtype=str)

        perception = environment.grid[min_x:max_x+1, min_y:max_y+1]
        local_grid[:perception.shape[0],:perception.shape[1]] = perception
        # local_grid = np.roll(local_grid, [local_x_start,local_y_start])
        if (self.x - self.vision_radius) < 0:
            local_grid = np.roll(local_grid, [3,0])
        if (self.y - self.vision_radius) < 0:
            local_grid = np.roll(local_grid, [0,1])
        # print(local_grid)  
        # print(self.name)
        return local_grid

    def decide(self, perception):

        # print(perception)
        # print(perception.flatten())
        # philia = np.zeros((1,2))
        # phobia = np.zeros((1,2))
        # for count, elem in enumerate(perception.flatten()):
            #     philia  += self.points[count]
            # if elem == 'X':
            #     phobia  += self.points[count]
            # print(self.name,elem)
        # print("philia",philia,'\n')
        # print("phobia",phobia)
        l,r,u,d,s = 0,0,0,0,0
        strand = np.zeros((1,5))
        for count, elem in enumerate(perception.flatten()):
            i,j = divmod(count,3)
            # print(i,j,'ij')
            probablities = self.gene[i,j,:,:]
            # print(elem)
            # print(probablities,"prob")
            # print(self.gene)
            # exit()
            if elem == 'F':
                strand += probablities[0]
            elif elem == 'X':
                strand += probablities[1]
            else:
                strand += probablities[2]
        strand /= strand.sum()
        # print(strand[0],"strand")
        return np.random.choice(['u', 'd', 'l', 'r', 's'],1,strand[0].tolist())  # Default random move

    def act(self, action):
        if action == 'u' and self.x > 0:
            self.x -= 1
        elif action == 'd' and self.x < self.grid_size - 1:
            self.x += 1
        elif action == 'l' and self.y > 0:
            self.y -= 1
        elif action == 'r' and self.y < self.grid_size - 1:
            self.y += 1
        else:
            self.fitness -= 5

    def update_fitness(self, environment):
        if environment.grid[self.x, self.y] == 'F':
            self.food_collected += 1
            self.fitness += 20
            # environment.grid[self.x, self.y] = ' '  # Remove the food from the grid
        if environment.grid[self.x, self.y] == 'X':
            # self.food_collected += 1
            self.fitness -= 10
            # environment.grid[self.x, self.y] = ' '  # Remove the obstacle from the grid

    def get_pos(self):
        return self.x,self.y
    
    def genome(self):
        return [_gene[1] for _gene in self.gene]
    #     return [_gene[1] for _gene in self.gene]
    # def __repr__(self) -> str:
    #     return ''.join([_gene[1] for _gene in self.gene])

def random_probabilities():
    a = np.random.random(5)
    a /= a.sum()
    return a

def simulate_generation(environment, num_steps):
    # for _ in range(num_steps):
    #     agent = environment.agents[0]
    #     if type(agent) is Agent:
    #         perception = agent.perceive(environment)
    #         action = agent.decide(perception)
    #         agent.act(action)
    #         agent.update_fitness(environment)
    #         # print(agent.x,agent.y)
    #     if type(agent) is tuple:
    #         perception = agent[0].perceive(environment)
    #         action = agent[0].decide(perception)
    #         agent[0].act(action)
    #         agent[0].update_fitness(environment)
    #         # print(agent[0].x,agent[0].y)
    #         break
    for _ in range(num_steps):
        # environment.display()
        for agent in environment.agents:
            if type(agent) is Agent:
                perception = agent.perceive(environment)
                action = agent.decide(perception)
                agent.act(action)
                agent.update_fitness(environment)
                # print(agent.x,agent.y)
            if type(agent) is tuple:
                perception = agent[0].perceive(environment)
                action = agent[0].decide(perception)
                agent[0].act(action)
                agent[0].update_fitness(environment)
                # print(agent[0].x,agent[0].y)

            # perception = agent.perceive(environment)
            # action = agent.decide(perception)
            # agent.act(action)
            # agent.update_fitness(environment)


def evaluate_population(environment):
    for agent in environment.agents:
        if type(agent) is Agent:
            print(f"Agent Fitness: {agent.fitness}, Food Collected: {agent.food_collected}")
        if type(agent) is tuple:
            print(f"Agent Fitness: {agent[0].fitness}, Food Collected: {agent[0].food_collected}")


def select_agents(agents, num_to_select):
    if type(agents[0]) is Agent:
        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
    if type(agents[0]) is tuple:
        agents = sorted(agents, key=lambda x: x[0].fitness, reverse=True)
    # agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
    return agents[:num_to_select]


def crossover(parent1, parent2):
    crossover_point = random.randrange(35,101,5)
    if type(parent1) is Agent:
        # crossover_point = random.randint(1, len(parent1.gene) - 1)
        child_gene = np.concatenate((parent1.gene.flatten()[:crossover_point] , parent2.gene.flatten()[crossover_point:]))
        child_gene = child_gene.reshape(3,3,3,5)
        return Agent(random.randint(0, parent1.grid_size - 1), random.randint(0, parent1.grid_size - 1), parent1.grid_size, vision_radius=parent1.vision_radius), child_gene
    if type(parent1) is tuple:
        # crossover_point = random.randint(1, len(parent1[0].gene) - 1)
        child_gene = np.concatenate((parent1[0].gene.flatten()[:crossover_point] , parent2[0].gene.flatten()[crossover_point:]))
        child_gene = child_gene.reshape(3,3,3,5)
        return Agent(random.randint(0, parent1[0].grid_size - 1), random.randint(0, parent1[0].grid_size - 1), parent1[0].grid_size, vision_radius=parent1[0].vision_radius), child_gene


def mutate(agent, mutation_rate=0.05):
    # print('Agenty',agent,'\n')
    if random.random() < mutation_rate:
        # mutation_point = random.randint(0, len(agent[0].gene) - 1)
        # actions = ['u', 'd', 'l', 'r', 's']
        # agent[0].gene[mutation_point] = (mutation_point, random.choice(actions))
        # TODO implement mutation logic (change probablity by a rate ALPHA)
        x,y,z = random.randint(0,2), random.randint(0,2), random.randint(0,2)
        print(agent[0].gene[x,y])
        exit()


def next_generation(environment, num_to_select, mutation_rate):
    selected_agents = select_agents(environment.agents, num_to_select)
    # print(selected_agents)
    new_agents = []

    while len(new_agents) < len(environment.agents):
        parent1, parent2 = random.sample(selected_agents, 2)
        child = crossover(parent1, parent2)
        mutate(child, mutation_rate)
        new_agents.append(child)

    environment.agents = new_agents
    # for agent in new_agents:
        # print(agent[0].genome())   

def main():
    size = 18  # Size of the grid (NxN)
    num_obstacles = 20
    num_food = 60
    num_agents = 40
    num_generations = 1000
    num_steps_per_generation = 50

    environment = Environment(size, num_obstacles, num_food, num_agents)
    # for agent in environment.agents:
    #     print(agent.name)
    #     agent.perceive(environment)
        

    for generation in range(num_generations):
        # next_generation(environment, num_to_select=5, mutation_rate=0.05)
        print(f"Generation {generation + 1}")
        # environment.display()
        simulate_generation(environment, num_steps_per_generation)
        evaluate_population(environment)
        next_generation(environment, num_to_select=5, mutation_rate=0.05)
        # environment.display()

if __name__ == "__main__":
    main()
