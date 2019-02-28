# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:11:46 2019

@author: iasedric
"""
# Keras is to be imported only once in the beggining.
'''
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
'''

import pygame
import random
import pandas as pd
import heapq
import numpy as np
from pygame.math import Vector2
from math import cos, sin, tan, pi, radians, degrees, copysign, sqrt


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

PROBA = 0.85
DELTA = 0.5

# Loading sprites
bg_sprite = pygame.image.load('Circuit.png')
car_image = pygame.image.load('Car.png')

'''
# Creating classifiers.
## TEMPORARY. Move to Car class.

classifier = [0] * NUMBER_MODELS

for i in range(0,NUMBER_MODELS):
    # Initialising the ANN
    classifier[i] = Sequential()
    # Adding the input layer and the first hidden layer
    classifier[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
    # Adding the output layer
    classifier[i].add(Dense(output_dim = 2, kernel_initializer='random_uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier[i].compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 


# Creating a holder to remember the 10 best models
best_classifiers = [0] * 10

for i in range(0,10):
    # Initialising the ANN
    best_classifiers[i] = Sequential()
    # Adding the input layer and the first hidden layer
    best_classifiers[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
    # Adding the output layer
    best_classifiers[i].add(Dense(output_dim = 2, kernel_initializer='random_uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    best_classifiers[i].compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
'''


# Defining classes
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
        self.position.x = int(self.position.x)
        self.position.y = int(self.position.y)
        self.angle += degrees(angular_velocity) * dt



# Defining functions
def clean_car(car_image, road_color):
    car_copy = car_image.copy()
    for i in range(0,CAR_WIDTH):
        for j in range(0,CAR_LENGTH):
            if (car_copy.get_at((i,j))==(255, 255, 255, 255)):
                car_copy.set_at((i,j),(255, 255, 255, 0))
            elif (car_copy.get_at((i,j))== ROAD_COLOR):
                car_copy.set_at((i,j),(255, 255, 255, 0))
    return car_copy
  
# Cleaning the car    
car_image = clean_car(car_image,ROAD_COLOR)

  
def calculate_distances(position, a):
    
    #if (bg_sprite.get_at((int(car.position.x),int(car.position.y)))!=ROAD_COLOR):
    # Front stream
    for i in range(0, SCREEN_WIDTH):
        xf = int(position.x + i *cos(3/2*pi + radians(-a)))
        yf = int(position.y + i *sin(3/2*pi + radians(-a)))
        if (bg_sprite.get_at((xf,yf))!=ROAD_COLOR):
            break;
    
    df = sqrt((position.x-xf)**2 + (position.y-yf)**2)
    pygame.draw.circle(win, (255,0,0), (int(xf), int(yf)), 2)
    
    # Right stream
    for i in range(0, SCREEN_WIDTH):
        xr = int(position.x + i *cos(2*pi + radians(-a)))
        yr = int(position.y + i *sin(2*pi + radians(-a)))
        if (bg_sprite.get_at((xr,yr))!=ROAD_COLOR):
            break;
            
    dr = sqrt((position.x-xr)**2 + (position.y-yr)**2)        
    pygame.draw.circle(win, (255,0,0), (int(xr), int(yr)), 2)
    
    # Left stream
    for i in range(0, SCREEN_WIDTH):
        xl = int(position.x + i *cos(pi + radians(-a)))
        yl = int(position.y + i *sin(pi + radians(-a)))
        if (bg_sprite.get_at((xl,yl))!=ROAD_COLOR):
            break;
    
    dl = sqrt((position.x-xl)**2 + (position.y-yl)**2)        
    pygame.draw.circle(win, (255,0,0), (int(xl), int(yl)), 2)
    
    # Right 45deg stream
    for i in range(0, SCREEN_WIDTH):
        xdr = int(position.x + i *cos(1.75*pi + radians(-a)))
        ydr = int(position.y + i *sin(1.75*pi + radians(-a)))
        if (bg_sprite.get_at((xdr,ydr))!=ROAD_COLOR):
            break;
    
    ddr = sqrt((position.x-xdr)**2 + (position.y-ydr)**2)        
    pygame.draw.circle(win, (255,0,0), (int(xdr), int(ydr)), 2)
    
    # Left 45deg stream
    for i in range(0, SCREEN_WIDTH):
        xdl = int(position.x + i *cos(1.25*pi + radians(-a)))
        ydl = int(position.y + i *sin(1.25*pi + radians(-a)))
        if (bg_sprite.get_at((xdl,ydl))!=ROAD_COLOR):
            break;
    
    ddl = sqrt((position.x-xdl)**2 + (position.y-ydl)**2)        
    pygame.draw.circle(win, (255,0,0), (int(xdl), int(ydl)), 2)
    
    return df,dr,dl,ddr,ddl
    
    


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
    
    calculate_distances(car.position,car.angle)

    rotated = pygame.transform.rotate(car_image, car.angle)
    rect = rotated.get_rect()
    win.blit(rotated, Vector2(car.position)  - (rect.width / 2, rect.height / 2))
    
    #Collision detection
    if (bg_sprite.get_at((int(car.position.x),int(car.position.y)))!=ROAD_COLOR):
        print("Out of road")
   
    
            
    pygame.display.update()
    

pygame.quit()