import pygame as pg
import torch
import numpy as np
from torch import nn
from utils import COLS_COUNT,ROWS_COUNT,CELLH,CELLW

class map:
    def __init__(self):
        self.MapMtx = [[1 for _ in range(ROWS_COUNT)]for _ in range(COLS_COUNT)] #  0: OBSTACLE 1: EMPTY 2: STORE
        self.ressources = {}
    def InitMap_1(self):
        #external wall
        for i in range(COLS_COUNT):
            self.MapMtx[0][i] =0
            self.MapMtx[i][0] =0
            self.MapMtx[ROWS_COUNT-1][i] =0
            self.MapMtx[i][COLS_COUNT-1] =0
        #simple map with one obstacle in the middle:
        for i in range(ROWS_COUNT//3,2*ROWS_COUNT//3):
            for j in range(COLS_COUNT//3,2*COLS_COUNT//3):
                self.MapMtx[i][j] = 0
        #init store zone
        for i in range(10):
            for j in range(10):
                self.MapMtx[i][j] = 2 
        #init ressources:
        while(len(self.ressources)!=25):
            RandomCordinate = np.random.randint(size=(2,),low=0,high= CELLW*COLS_COUNT)
            if self.MapMtx[RandomCordinate[0]//CELLW][RandomCordinate[1]//CELLH] == 1:
                self.ressources[tuple(RandomCordinate)] = 1

    def Draw_Map(self,Screen: pg.display):
        # draw map
        for i in range(ROWS_COUNT):
            for j in range(COLS_COUNT):
                CellRect = pg.Rect(j*CELLW,i*CELLH,CELLW,CELLH)
                if self.MapMtx[i][j] == 0:
                    pg.draw.rect(Screen,"black",CellRect)
                elif self.MapMtx[i][j] == 2:
                    pg.draw.rect(Screen,"yellow",CellRect)   
        # draw ressources 
        for ressources in self.ressources:
           if self.ressources[ressources] == 1:
               pg.draw.rect(Screen,"red",pg.Rect(ressources[0],ressources[1],CELLW,CELLH))