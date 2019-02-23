# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:11:46 2019

@author: iasedric
"""

import pygame
import random
import pandas as pd
import heapq
import numpy as np


# Defining constants
#NUMBER_MODELS = 20

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
TEXT_ZONE = 100


CAR_INIT_X = 75
CAR_INIT_Y = 500
CAR_SPEED = 5
CAR_INIT_ANGLE = 0
CAR_LENGTH = 33
CAR_WIDTH = 19
CAR_MAX_STEER_ANGLE = 30
CAR_CUR_STEER_ANGLE = 0


ROAD_COLOR = (96, 96, 96, 255)

# Loading sprites
#pygame.display.set_caption("Flappy Bird")
bg_sprite = pygame.image.load('Circuit.png')

car = pygame.image.load('Car.png')


#car = pygame.Rect(CAR_INIT_X,CAR_INIT_Y,CAR_WIDTH,CAR_LENGTH)


def clean_car(car_image, road_color):
    car_copy = car_image.copy()
    for i in range(0,CAR_WIDTH):
        for j in range(0,CAR_LENGTH):
            if (car_copy.get_at((i,j))==(255, 255, 255, 255)):
                car_copy.set_at((i,j),(255, 255, 255, 0))
            elif (car_copy.get_at((i,j))== ROAD_COLOR):
                car_copy.set_at((i,j),(255, 255, 255, 0))
    return car_copy
  
    
car_im = clean_car(car,ROAD_COLOR)

class Car():
        
    def __init__(self):
        self.x = CAR_INIT_X
        self.y = CAR_INIT_Y
        self.vx = 0
        self.vy = CAR_SPEED
        self.a = CAR_INIT_ANGLE
        self.length = CAR_LENGTH
        self.width = CAR_WIDTH
        self.maxstangle = CAR_MAX_STEER_ANGLE 
        self.curstangle = CAR_CUR_STEER_ANGLE
    
    def move():
        print("moving")
        
    def turn_right():
        print("turning right")
        
    def turn_left():
        print("turning left")
        
        
    
    


### Main    
    
pygame.init()
pygame.font.init()


#Opening the screen
win = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT+TEXT_ZONE))   

run = True

while run:
    pygame.time.delay(100)
    

    win.fill((0,0,0))
    
    win.blit(bg_sprite, (0,0))
    
    win.blit(car_im, (CAR_INIT_X,CAR_INIT_Y))
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            
            run = False
            
    #keys = pygame.key.get_pressed()
    
    #if keys[pygame.K_LEFT]:
        
    #if keys[pygame.K_RIGHT]:
            
    #pygame.draw.rect(win, (34, 139, 34), car)
    
    
    #car_im = rot_center(car_im , 45)        
            
    pygame.display.update()
    

pygame.quit()