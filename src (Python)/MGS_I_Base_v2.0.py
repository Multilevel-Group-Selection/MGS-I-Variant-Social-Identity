import random
import math
import numpy as np
import matplotlib.pyplot as plt


# -----------
# PARAMETERS
# -----------

# Set size of social space.
ROWS, COLS = 22, 22

# Set initial conditions.
INITIAL_PROSOCIAL_FRACTION = .1
DENSITY = 0.7
PRESSURE = 1.06
SYNERGY = 2.4

# Set simulation length.
TICK_MAX = 200


# --------------------
# CLASSES & FUNCTIONS
# --------------------

class Agent():
    """Represents an agent in the simulation.
    
       agent ids start at 1
       agent locs are world coordinates
       agent policy is binary
       agent group_patience is real in [0, 1)
       agent policy_patience is real in [0, 1)
       agent group is a list of agents in agent's group
       agent score is the payoff from the agent's policy and group"""
    
    def __init__(self, agent_id, spot_loc, agent_policy, *,
                group_patience=0.5,
                policy_patience=0.5):
        self.id = agent_id
        self.loc = spot_loc 
        self.policy = agent_policy
        self.group_patience = group_patience
        self.policy_patience = policy_patience
        self.group = []
        self.score = 0.
        
    def __str__(self):
        return (f'{self.__class__.__name__}: '
                f'{self.id} is at '
                f'{self.loc} with policy value '
                f'{self.policy}, group patience '
                f'{self.group_patience}, policy patience '
                f'{self.policy_patience}, and current score '
                f'{self.score})')
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.id}, '
                f'{self.loc}, '
                f'{self.policy}, '
                f'{self.group_patience}, '
                f'{self.policy_patience}, '
                f'{self.score})')

    def update_score(self):
        """Updates agent's score based on its current group."""
        contribution = 0
        group_size = len(self.group)
        if group_size > 1:
            for groupmember in self.group:
                contribution += groupmember.policy
            self.score = (1 - self.policy) + SYNERGY * contribution / group_size
        else:
            self.score = 1
        
    def wants_to_change_group(self):
        """Returns True if agent decides to change groups."""
        if random.random() >= self.group_patience:
            return True
        else:
            return False
    
    def wants_to_change_policy(self):
        """Returns True if agent decides to change its policy."""
        if random.random() >= self.policy_patience:
            return True
        else:
            return False
    
    def change_policy(self):
        """Changes an agent's policy"""
        self.policy = (self.policy + 1) % 2
    

def setup(agents, world, population):
    """Creates agent population according to simulation parameters."""
    for agent_id in range(1, population + 1):
        spot = (random.randrange(ROWS - 1), random.randrange(COLS - 1))
        while world[spot]:
            spot = (random.randrange(ROWS - 1), random.randrange(COLS - 1))
        world[spot] = agent_id
        agents.append(Agent(agent_id, spot, 0))
    
    initialprosocial = int(population * INITIAL_PROSOCIAL_FRACTION)
    for agent in agents[:initialprosocial]:
        agent.policy = 1


def update_groups(agents, field):
    """Updates each agent's group list."""
    population = len(agents)
    for agent in agents:
        agent.group = [agent]
    for pagent in agents:
        for nagent in agents[pagent.id:population]:
            if math.dist(pagent.loc, nagent.loc) < 1.5:
                pagent.group.append(nagent)
                nagent.group.append(pagent)
    

def count_agents(value, agentset):
    """Counts agents in agentset with with value as their policy."""
    count = 0
    for agent in agentset:
        if value == agent.policy:
            count += 1
    return count


def empty_spots(world):
    """Generator of empty spots in world."""
    for x in range(ROWS):
        for y in range (COLS):
            if world[x][y] == 0:
                yield((x, y))
                

# ----------
# SIMULATION
# ----------                        

def simulate():
    random.seed()

    # Prepare empty lists.
    time, prosocial_fraction = [], []
    world = np.zeros((ROWS, COLS)) 
    agents = []

    # Calculate population size is the product of size of the social space and density.
    population = int(DENSITY * COLS * ROWS)

    # Setup the world.
    setup(agents, world, population)

    # Set up time and book keeping
    tick = 0
    time.append(tick)
    prosocial_fraction.append(count_agents(1, agents) / population)

    while True:
        tick += 1
        
        # Update group membership for all agents.
        update_groups(agents, world)
        
        # Calculate agent satisfaction for all agents.
        for agent in agents:
            agent.update_score()
    
        # Check agent satisfaction and allow all unsatisfied agents to choose actions.
        agents_moving = []
        agents_switching = []
        unsatisfied = 0
        for agent in agents:
            if agent.score < PRESSURE:
                unsatisfied += 1
                if agent.wants_to_change_group():
                    agents_moving.append(agent)
                if agent.wants_to_change_policy():
                    agents_switching.append(agent)
    
        # End the simulation if all agents are satisfied.               
        if unsatisfied == 0:
            break
            
        # Relocate all agents that are moving.
    
        # Start by creating a list of open spots, then append spots of all agents who will move.
        available_spots = list(empty_spots(world))
        for agent in agents_moving:
            available_spots.append(agent.loc)
            
        # Find each agent a new spot then move the agent    
        for agent in agents_moving:
            old_spot = agent.loc
            new_spot = random.choice(available_spots)
            while old_spot == new_spot:
                new_spot = random.choice(available_spots)
                
            # Move the agent and reduce the set of available spots           
            world[old_spot] = 0
            world[new_spot] = agent.id
            agent.loc = new_spot     
            available_spots.remove(new_spot)
    
        # Update policies for all agents that are changing.
        for agent in agents_switching:
            agent.change_policy()
    
            
        # Perform end of tick book keeping.
        time.append(tick)
        prosocial_fraction.append(count_agents(1, agents) / population)
    
        # End the simulation if out of time. 
        if tick > TICK_MAX:
            break
    return (time, prosocial_fraction)

if __name__ == "__main__":
    (time, prosocial_fraction) = simulate()
    
    # PLOTTING
    plt.plot(time, prosocial_fraction)
    plt.ylim(ymin=0., ymax=1.)
    plt.title('Contributors')
    plt.xlabel('ticks')
    plt.ylabel('% of population')
    plt.show()
