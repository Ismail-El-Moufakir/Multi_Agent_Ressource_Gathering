import pygame as pg
import map as mp
import matplotlib.pyplot as plt
import numpy as np
import time
from Agent import Agent,GenAlgo
from utils import GENERATION_COUNT,POPULATION_SIZE

#torch init
scores =[]
#init the Simulation
Screen = pg.display.set_mode(size=(600,600))
gen = GenAlgo(10,5,5,0.1,0.6,0.4)
gen.init_population(Agent)
for i in range(GENERATION_COUNT):
    is_running = True
    map = mp.map()
    map.InitMap_1()
    Timer = 0
    while is_running and Timer<100:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                is_running = False
        Screen.fill("white")
        for agent in gen.population:
            agent.Action(map)
            agent.move(map)
            agent.draw_Agent(Screen)
        map.Draw_Map(Screen)
        pg.display.flip()
        #tempo th see what happen
        Timer += 1
    print("Simulation",i,"Ended----------------------")
    
    All_fitnessScore = [gen.fitness_function(ag) for ag in gen.population]
    print(f"firness function of agents {All_fitnessScore}")
    scores.append(np.mean(All_fitnessScore))
    gen.new_population()
    #a simple plotting at the end of the mean of the fitrness function
    
pg.quit()
plt.plot(scores)
plt.show()


