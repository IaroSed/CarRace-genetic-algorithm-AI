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
from pygame.math import Vector2
from math import tan, radians, degrees, copysign, sqrt


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
CAR_MAX_STEER_ANGLE = 60



ROAD_COLOR = (96, 96, 96, 255)

# Loading sprites
#pygame.display.set_caption("Flappy Bird")
bg_sprite = pygame.image.load('Circuit.png')

car_image = pygame.image.load('Car.png')


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
  
    
car_image = clean_car(car_image,ROAD_COLOR)


class Car():
        
    def __init__(self):
        self.position = Vector2(CAR_INIT_X,CAR_INIT_Y)
        self.velocity = Vector2(0.0, -1*CAR_SPEED)
        self.angle = CAR_INIT_ANGLE
        self.length = CAR_LENGTH
        #self.width = CAR_WIDTH
        #self.max_acceleration = 0
        self.max_steering = CAR_MAX_STEER_ANGLE 
        
        self.acceleration = 0.0
        self.steering = 0.0

    
    def update(self, dt):
        #Velocity is constant for the moment
        #self.velocity += (self.acceleration * dt, 0)
        
        if self.steering:
            turning_radius = self.length / tan(radians(self.steering))
            angular_velocity = sqrt(self.velocity.x**2 + self.velocity.y**2) / turning_radius

        else:
            angular_velocity = 0
            
        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt


        
        
    
car = Car()    


### Main    
    
pygame.init()
pygame.font.init()
#start = pygame.time.get_ticks()

#Opening the screen
win = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT+TEXT_ZONE))   

run = True

while run:
    
    pygame.time.delay(100)
    
    #update = pygame.time.get_ticks() 
    #dt = (update - start)/100
    #start = update
    
    dt = 1

    win.fill((0,0,0))
    
    win.blit(bg_sprite, (0,0))

    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            
            run = False
            
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_LEFT]:
        car.steering += 30 * dt
    elif keys[pygame.K_RIGHT]:
        car.steering -= 30 * dt
    else:
        car.steering = 0

    car.steering = max(-car.max_steering, min(car.steering, car.max_steering))
    
    car.update(dt)

    rotated = pygame.transform.rotate(car_image, car.angle)
    rect = rotated.get_rect()

    win.blit(rotated, Vector2(car.position)  - (rect.width / 2, rect.height / 2))
    
    #Collision detection
    if (bg_sprite.get_at((int(car.position.x),int(car.position.y)))!=ROAD_COLOR):
        print("Out of road")
   
            
    pygame.display.update()
    

pygame.quit()