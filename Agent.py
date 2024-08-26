import pygame as pg
import torch
from torch import nn
from utils import COLS_COUNT, ROWS_COUNT, CELLH, CELLW
from utils import POPULATION_SIZE, GRAB_REWARD, OBSTACLE_PENALTY, TOURNATENT_SIZE,CROSSOVER_RATE
import map as mp
import numpy as np
import time

class Agent(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        # Agent Coordinate:
        self.x = x
        self.y = y
        self.speed = [2, 2]
        self.ObstacleHit = 0
        self.grabbed = 0
        self.grabbedRessources = None
        self.ressourcesDelivered = 0
        # NN Agent definition
        self.Linear_Relu_Stack = nn.Sequential(
            nn.Linear(11, 50),
            nn.Sigmoid(),
            nn.Linear(50, 50),
            nn.Sigmoid(),
            nn.Linear(50, 2),
        )
    
    def forward(self, inputs):
        return self.Linear_Relu_Stack(inputs)
    
    def Action(self, map: mp.map):
        # Find all walls close to Agent
        IntXpos = self.x // CELLW
        IntYpos = self.y // CELLH
        AllWalls = [0 for _ in range(8)]

        k = 0

        for i in range(max(0, IntYpos - 1), min(IntYpos + 2, ROWS_COUNT)):
            for j in range(max(0, IntXpos - 1), min(IntXpos + 2, COLS_COUNT)):
                if i == IntYpos and j == IntXpos:
                    continue
                if map.MapMtx[i][j] == 0:
                    AllWalls[k] = 1
                k += 1
        
        # Checking collision with resources and grabbing them:
        GRAB = 0
        for ressource in map.ressources:
            ressourceRect = pg.Rect(ressource[0], ressource[1], 10, 10)
            agentRect = pg.Rect(self.x, self.y, 20, 20)
            if agentRect.colliderect(ressourceRect) and not self.grabbed:
                GRAB = 1
                self.grabbedRessources = [ressource[0], ressource[1]]
                self.grabbed = 1
                map.ressources[ressource] = 0
                
        # Defining the input for the neural network 
        # [W1, W2, W3, W4, W5, W6, W7, W8, x, y, GRAB]
        inputs = torch.tensor(AllWalls + [(self.x // CELLW) / COLS_COUNT, (self.y // CELLH) / ROWS_COUNT, GRAB], dtype=torch.float32)
        logits = self.forward(inputs)
        
        # Determine the action based on the neural network output
        action = [0, 0, GRAB]
        action[0] = self.speed[0] if logits[0] < 0 else -self.speed[0]
        action[1] = self.speed[1] if logits[1] < 0 else -self.speed[1]
    
        return action

    def move(self,map: mp.map):
        action  = self.Action(map)
        
        AllWalls = []
        IntXpos = self.x // CELLW
        IntYpos = self.y // CELLH
        #finding all walls in close to agent 
        for i in range(max(0, IntYpos - 1), min(IntYpos + 2, ROWS_COUNT)):
            for j in range(max(0, IntXpos - 1), min(IntXpos + 2, COLS_COUNT)):
                if map.MapMtx[i][j] == 0 or map.MapMtx[i][j] == 2:
                    AllWalls.append((j,i))
        agentRect = pg.Rect(self.x,self.y,20,20)
        #checking collision to prevent going beyond boundaries
        for wall in AllWalls:
            wallRect = pg.Rect(wall[0]*CELLW,wall[1]*CELLH,CELLW,CELLH)
            if agentRect.colliderect(wallRect):
                if agentRect.colliderect(wallRect):
                    if action[0] > 0 and agentRect.right > wallRect.left:  # Moving right
                        action[0] = 0
                       
                        self.ObstacleHit += 1
                    elif action[0] < 0 and agentRect.left < wallRect.right:  # Moving left
                        action[0] = 0
                        
                        self.ObstacleHit += 1
                    if action[1] > 0 and agentRect.bottom > wallRect.top:  # Moving down
                        action[1] = 0
                        
                        self.ObstacleHit += 1
                    elif action[1] < 0 and agentRect.top < wallRect.bottom:  # Moving up
                        action[1] = 0
                        
                        self.ObstacleHit += 1
        #checking grabbed ressources
        if self.grabbed == 1:
                self.grabbedRessources[0] = agentRect.right
                self.grabbedRessources[1] = agentRect.bottom
        #check if agent deliver ressources
        if self.grabbed == 1:
            for i in range(0,COLS_COUNT):
                        for j in range(0,ROWS_COUNT):
                            if map.MapMtx[j][i] == 2:
                                wallRect = pg.Rect(i*CELLW,j*CELLH,CELLW,CELLH)
                                if agentRect.colliderect(wallRect):
                                        self.grabbedRessources = None
                                        self.ressourcesDelivered += 1
                                        self.grabbed = 0
                        
           
                            

        self.x += action[0]
        self.y += action[1]
        
    def draw_Agent(self,Screen: pg.display):
        pg.draw.rect(Screen,"aquamarine3",pg.Rect(self.x,self.y,20,20))
        if self.grabbedRessources != None:
            pg.draw.rect(Screen,"red",pg.Rect(self.grabbedRessources[0],self.grabbedRessources[1],10,10))



class GenAlgo:
    def __init__(self, population_size, tournament_size, grab_reward, obstacle_penalty, crossover_rate, mutation_rate):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.grab_reward = grab_reward
        self.obstacle_penalty = obstacle_penalty
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []

    def init_population(self, Agent):
        for i in range(self.population_size):
            agent = Agent(np.random.randint(0, 600), np.random.randint(0, 600))
            self.population.append(agent)

    def fitness_function(self, agent):
        return agent.grabbed * self.grab_reward - agent.ObstacleHit * self.obstacle_penalty + 10*agent.ressourcesDelivered

    def selection(self):
        selected = []
        while len(selected) < self.population_size:
            candidates = {}
            for _ in range(self.tournament_size):
                random_agent = np.random.choice(self.population)
                candidates[random_agent] = self.fitness_function(random_agent)
            selected.append(max(candidates, key=candidates.get))
        return selected

    def crossover(self, parent1, parent2, Agent):
        child = Agent(np.random.randint(0, 600), np.random.randint(0, 600))
        child_params = list(child.parameters())
        parent1_params = list(parent1.parameters())
        parent2_params = list(parent2.parameters())

        for i in range(len(child_params)):
            param = child_params[i]
            parent1_param = parent1_params[i]
            parent2_param = parent2_params[i]

            if parent1_param.shape == parent2_param.shape:
                split_point = np.random.randint(0, parent1_param.shape[0])
                new_param = torch.cat((parent1_param[:split_point], parent2_param[split_point:]), dim=0)
                param.data = new_param
            else:
                raise ValueError(f"Parameter shape mismatch: {parent1_param.shape} vs {parent2_param.shape}")

        return child

    def mutation(self, parent, Agent):
        child = Agent(np.random.randint(0, 600), np.random.randint(0, 600))
        parent_params = list(parent.parameters())
        child_params = list(child.parameters())

        for i in range(len(child_params)):
            parent_param = parent_params[i]
            param = child_params[i]

            noise = torch.normal(0, self.mutation_rate, size=parent_param.shape)
            param.data = parent_param.data + noise

        return child

    def new_population(self):
        selected = self.selection()
        new_population = []

        while len(new_population) < self.population_size:
            np.random.shuffle(selected)
            random_factor = np.random.rand()

            if random_factor < self.crossover_rate:
                parent1, parent2 = selected[0], selected[1]
                child = self.crossover(parent1, parent2, Agent)
            else:
                parent = np.random.choice(selected)
                child = self.mutation(parent, Agent)

            new_population.append(child)

        self.population = new_population

    def run_generation(self):
        self.new_population()




            
        
        
        

            

        

   



            
        