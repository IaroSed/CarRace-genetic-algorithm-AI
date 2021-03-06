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
NUMBER_MODELS = 20
classifier = [0] * NUMBER_MODELS
for i in range(0,NUMBER_MODELS):
    # Initialising the ANN
    classifier[i] = Sequential()
    # Adding the input layer and the first hidden layer
    classifier[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
    # Adding second hidden layer
    classifier[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
    # Adding third hidden layer
    classifier[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
    # Adding the output layer
    classifier[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier[i].compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
# Creating a holder to remember the 10 best models
best_classifiers = [0] * 10
for i in range(0,10):
    # Initialising the ANN
    best_classifiers[i] = Sequential()
    # Adding the input layer and the first hidden layer
    best_classifiers[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
    # Adding second hidden layer
    best_classifiers[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
    # Adding third hidden layer
    best_classifiers[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
    # Adding the output layer
    best_classifiers[i].add(Dense(output_dim = 5, kernel_initializer='random_uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    best_classifiers[i].compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
'''

import pygame
import random
import pandas as pd
import heapq
import numpy as np
from pygame.math import Vector2
from math import cos, sin, tan, pi, radians, degrees, copysign, sqrt


# Defining constants
NUMBER_MODELS = 20

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
TEXT_ZONE = 100


CAR_INIT_X = 75
CAR_INIT_Y = 500
CAR_SPEED = 6
CAR_INIT_ANGLE = 0
CAR_LENGTH = 33
CAR_WIDTH = 19
CAR_MAX_STEER_ANGLE = 30
CAR_MAX_ACCELERATION = 5
CAR_MAX_VELOCITY = 100

CAR_BRAKE_DECELERATION = 2
CAR_FREE_DECELERATION = 1

ROAD_COLOR = (96, 96, 96, 255)

PROBA = 0.85
DELTA = 0.5

# Loading sprites
bg_sprite = pygame.image.load('Circuit.png')
car_image = pygame.image.load('Car.png')

Exit_RS = open("ResultsDriving.txt", 'w',encoding='utf-8')

# Defining classes
class Car():
        
    def __init__(self):
        self.position = Vector2(CAR_INIT_X,CAR_INIT_Y)
        self.velocity = Vector2(0.0, -1*CAR_SPEED)
        self.angle = CAR_INIT_ANGLE
        self.length = CAR_LENGTH
        self.max_acceleration = CAR_MAX_ACCELERATION
        self.max_steering = CAR_MAX_STEER_ANGLE
        self.max_velocity = CAR_MAX_VELOCITY

        self.brake_deceleration = CAR_BRAKE_DECELERATION
        self.free_deceleration = CAR_FREE_DECELERATION
        
        self.acceleration = 0.0
        self.steering = 0.0
        
        self.alive = True
        self.fitness = 0
        self.index = 0

    
    def update(self, dt):
        
        if self.velocity.y < 0:
            self.velocity -= (0, self.acceleration * dt)
        else:
            self.velocity += (0, self.acceleration * dt)
        
        self.velocity.y = copysign(min(abs(self.velocity.y), self.max_velocity),self.velocity.y)
        
        if self.steering:
            turning_radius = self.length / tan(radians(self.steering))
            angular_velocity = sqrt(self.velocity.x**2 + self.velocity.y**2) / turning_radius

        else:
            angular_velocity = 0
        
        
        self.position += self.velocity.rotate(-self.angle) * dt
        self.position.x = int(self.position.x)
        self.position.y = int(self.position.y)
        self.angle += degrees(angular_velocity) * dt
        
        
        #Updating the fitness
        if self.velocity.y != 0:
            self.fitness +=  abs(self.velocity.y)
        



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
        try:
            if (bg_sprite.get_at((xf,yf))!=ROAD_COLOR):
                break;
        except:
            xf = position.x
            yf = position.y
    
    df = sqrt((position.x-xf)**2 + (position.y-yf)**2)
    #pygame.draw.circle(win, (255,0,0), (int(xf), int(yf)), 2)
    
    # Right stream
    for i in range(0, SCREEN_WIDTH):
        xr = int(position.x + i *cos(2*pi + radians(-a)))
        yr = int(position.y + i *sin(2*pi + radians(-a)))
        try:
            if (bg_sprite.get_at((xr,yr))!=ROAD_COLOR):
                break;
        except:
            xr = position.x
            yr = position.y
            
    dr = sqrt((position.x-xr)**2 + (position.y-yr)**2)        
    #pygame.draw.circle(win, (255,0,0), (int(xr), int(yr)), 2)
    
    # Left stream
    for i in range(0, SCREEN_WIDTH):
        xl = int(position.x + i *cos(pi + radians(-a)))
        yl = int(position.y + i *sin(pi + radians(-a)))
        try:
            if (bg_sprite.get_at((xl,yl))!=ROAD_COLOR):
                break;
        except:
            xl = position.x
            yl = position.y
    
    dl = sqrt((position.x-xl)**2 + (position.y-yl)**2)        
    #pygame.draw.circle(win, (255,0,0), (int(xl), int(yl)), 2)
    
    # Right 45deg stream
    for i in range(0, SCREEN_WIDTH):
        xdr = int(position.x + i *cos(1.75*pi + radians(-a)))
        ydr = int(position.y + i *sin(1.75*pi + radians(-a)))
        try:
            if (bg_sprite.get_at((xdr,ydr))!=ROAD_COLOR):
                break;
        except:
            xdr = position.x
            ydr = position.y
    
    ddr = sqrt((position.x-xdr)**2 + (position.y-ydr)**2)        
    #pygame.draw.circle(win, (255,0,0), (int(xdr), int(ydr)), 2)
    
    # Left 45deg stream
    for i in range(0, SCREEN_WIDTH):
        xdl = int(position.x + i *cos(1.25*pi + radians(-a)))
        ydl = int(position.y + i *sin(1.25*pi + radians(-a)))
        try:
            if (bg_sprite.get_at((xdl,ydl))!=ROAD_COLOR):
                break;
        except:
            xdl = position.x
            ydl = position.y
    
    ddl = sqrt((position.x-xdl)**2 + (position.y-ydl)**2)        
    #pygame.draw.circle(win, (255,0,0), (int(xdl), int(ydl)), 2)
    
    return df,dr,dl,ddr,ddl


    
def crossover(fitness):
    

    best_fitness_temp = [0] * 30
    #print("1: " + str(best_classifiers[0].get_weights()[0][0]))

    best_fitness_temp = best_fitness + fitness

        
    best_parents_int = []
    best_parents_int = heapq.nlargest(30, enumerate(best_fitness_temp), key=lambda x: x[1])
    
    print("The last best is: " + str(heapq.nlargest(1, enumerate(best_fitness_temp), key=lambda x: x[1])[0][1]) + " the best is: " + str(best_fitness[0]))
    
    # Updating the best classifiers
    for i in range(0,10):
        best_classifiers[i].set_weights((best_classifiers+classifier)[best_parents_int[i][0]].get_weights())
        best_fitness[i] = best_fitness_temp[best_parents_int[i][0]]
        #Exit_RS.write(str(best_parents_int[i][0]) + "^")
    #Exit_RS.write(str(Generation) + "^" + str(best_fitness[0]))
    #Exit_RS.write("\n")


    #Replacing the first 5 classifiers by the best and no mutation 
    for i in range(0,5):
        classifier[i].set_weights(best_classifiers[i].get_weights())

    # Keeping some bad ones for diversity   
    for i in range(5,10):
        classifier[i].set_weights((best_classifiers+classifier)[best_parents_int[i+5][0]].get_weights())

    #Replacing the first 5 classifiers by the best and mutation authorized
    for i in range(10,15):
        classifier[i].set_weights(best_classifiers[i-10].get_weights())
        
    # Creating some cross parents
    CPw = [0] * 5
    for i in range(15,20):
        classifier[i].set_weights(best_classifiers[i-15].get_weights()) 
        CPw[i-15] = best_classifiers[i-15].get_weights()[2]
 
    for i in range(15,20):
        classifier[i].set_weights([best_classifiers[i-15].get_weights()[0], best_classifiers[i-15].get_weights()[1], CPw[14-i] , best_classifiers[i-15].get_weights()[3]])  
        

        
def mutate():
    
    
    #Introducing mutations: with a probability of (100% - PROBA) each weight can be changed by a number between -DELTA to +DELTA.
    for i in range(5,NUMBER_MODELS):
        
        #print(i)
        
        A1 = classifier[i].get_weights()[0][0]
        B1 = classifier[i].get_weights()[0][1]
        C1 = classifier[i].get_weights()[0][2]
        D1 = classifier[i].get_weights()[0][3]
        E1 = classifier[i].get_weights()[0][4]
        
        A2 = classifier[i].get_weights()[2][0]
        B2 = classifier[i].get_weights()[2][1]
        C2 = classifier[i].get_weights()[2][2]
        D2 = classifier[i].get_weights()[2][3]
        E2 = classifier[i].get_weights()[2][4]
        
        A3 = classifier[i].get_weights()[4][0]
        B3 = classifier[i].get_weights()[4][1]
        C3 = classifier[i].get_weights()[4][2]
        D3 = classifier[i].get_weights()[4][3]
        E3 = classifier[i].get_weights()[4][4]
        
        
        A4 = classifier[i].get_weights()[6][0]
        B4 = classifier[i].get_weights()[6][1]
        C4 = classifier[i].get_weights()[6][2]
        D4 = classifier[i].get_weights()[6][3]
        E4 = classifier[i].get_weights()[6][4]
        
        
        Z1 = classifier[i].get_weights()[1]
        Z2 = classifier[i].get_weights()[3]
        Z3 = classifier[i].get_weights()[5]
        Z4 = classifier[i].get_weights()[7]

        #number_changes = 0
        
        for j in range(0,5):
            
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                A1[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                B1[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                C1[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                D1[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                E1[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                A2[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                B2[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                C2[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                D2[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                E2[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                A3[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                B3[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                C3[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                D3[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                E3[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
                
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                A4[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                B4[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                C4[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                D4[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                E4[j] += change
                classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3, np.array([list(A4),list(B4),list(C4),list(D4),list(E4)]), Z4])
                #classifier[i].set_weights([np.array([list(A1),list(B1),list(C1),list(D1),list(E1)]), Z1, np.array([list(A2),list(B2),list(C2),list(D2),list(E2)]), Z2, np.array([list(A3),list(B3),list(C3),list(D3),list(E3)]), Z3])
                #number_changes += 1
            
    print("7: " + str(best_classifiers[0].get_weights()[0][0]))    


## End of functions


### Main    
    
pygame.init()
pygame.font.init()
#start = pygame.time.get_ticks()

#Opening the screen
win = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT+TEXT_ZONE))

car = list(range(0,NUMBER_MODELS))
X = list(range(0,NUMBER_MODELS))
Turn_right = list(range(0,NUMBER_MODELS))
Turn_left = list(range(0,NUMBER_MODELS))
Acceleration_up = list(range(0,NUMBER_MODELS)) 
Acceleration_down = list(range(0,NUMBER_MODELS)) 
Brakes = list(range(0,NUMBER_MODELS)) 
        
for i in range(0,NUMBER_MODELS):
    car[i] = Car()
    X[i] = pd.DataFrame([[0 , 0, 0, 0, 0]], columns=['DistanceFront','DistanceRight', 'DistanceLeft', 'DistanceDiagRight', 'DistanceDiagLeft'])


alive = []
fitness = []


# Controlling the direction of movement
Rect_list = [pygame.Rect(20,200,100,10), pygame.Rect(270,170,100,10), pygame.Rect(530,170,100,10), pygame.Rect(790,200,100,10), pygame.Rect(270,515,130,10), pygame.Rect(790,670,80,10), pygame.Rect(510,760,80,10), pygame.Rect(30,600,80,10)]


Generation = 1

myfont = pygame.font.SysFont('Comic Sans MS', 20)

best_fitness = [0] * 10
the_best_fitness = 0

time_since_start = 0

first_car_index = 0

run = True

while run:
    
    #pygame.time.delay(5)
    
    generation_info = myfont.render('Generation ' + str(Generation) + " Best fitness: " + str(best_fitness[0]) , False, (255, 255, 255))
    
    
    dt = 1
    
    time_since_start += dt
    
    if time_since_start % 100 == 0:
        print(time_since_start)

    win.fill((0,0,0))
    
    win.blit(bg_sprite, (0,0))

    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            
            run = False
    
    for i in range(0,NUMBER_MODELS):    
        
        rotated = pygame.transform.rotate(car_image, car[i].angle)
        car_rect = rotated.get_rect()
        
        win.blit(rotated, Vector2(car[i].position)  - (car_rect.width / 2, car_rect.height / 2))
        
        if car[i].alive:
            
            car[i].update(dt)
            
            # Calculating the distances around the car - inputs of the ANN model
            X[i].iloc[0,0], X[i].iloc[0,1],X[i].iloc[0,2],X[i].iloc[0,3],X[i].iloc[0,4] = calculate_distances(car[i].position,car[i].angle)
     
           
            # Predicting the Test set results:
            Turn_right[i] = classifier[i].predict(X[i])[0][0]
            Turn_left[i] = classifier[i].predict(X[i])[0][1]
            Acceleration_up[i] = classifier[i].predict(X[i])[0][2]
            Acceleration_down[i] = classifier[i].predict(X[i])[0][3]
            Brakes[i] = classifier[i].predict(X[i])[0][4]
            
            Turn_right[i] = (Turn_right[i] > 0.5)
            Turn_left[i] = (Turn_left[i] > 0.5)
            Acceleration_up[i] = (Acceleration_up[i] > 0.5)
            Acceleration_down[i] = (Acceleration_down[i] > 0.5)
            Brakes[i] = (Brakes[i] > 0.5)

                
            
            if Acceleration_up[i]:
    
                car[i].acceleration += 1 * dt
                
            if Acceleration_down[i]:
                
                car[i].acceleration -= 1 * dt
            
            if Brakes[i]:
    
                car[i].acceleration = 0
                
                if car[i].velocity.y < 0:
                    car[i].velocity.y += car[i].brake_deceleration
                    car[i].velocity.y = min(0, max(car[i].velocity.y, -1*car[i].max_velocity))
                 
            if (Acceleration_up[i] == False and Acceleration_down[i] == False and Brakes[i] == False):
    
                car[i].acceleration = 0
                
                if car[i].velocity.y < 0:
                    car[i].velocity.y += car[i].free_deceleration
                    car[i].velocity.y = min(0, max(car[i].velocity.y, -1*car[i].max_velocity))
    
            car[i].acceleration = max(0, min(car[i].acceleration, car[i].max_acceleration))
           
            
            if Turn_left[i]:
                car[i].steering += 30 * dt
            elif Turn_right[i]:
                car[i].steering -= 30 * dt
            else:
                car[i].steering = 0
                
            car[i].steering = max(-car[i].max_steering, min(car[i].steering, car[i].max_steering))
            
            #if i == first_car_index:
            #print("Generation:", Generation, "Fitness:", car[i].fitness, "Acceleration up:",Acceleration_up[i],"Acceleration down:",Acceleration_down[i], "Brakes", Brakes[i], "Car Velocity:", car[i].velocity, "Car acceleration:", car[i].acceleration)
            Exit_RS.write("Generation:" + '^' + str(Generation)  + '^' +  "Car number" + '^' + str(i)  + '^' +  "Fitness:" + '^' +  str(car[i].fitness) + '^' +  "Acceleration up:" + '^' + str(Acceleration_up[i]) + '^' + "Acceleration down:" + '^' + str(Acceleration_down[i])  + '^' +  "Brakes" + '^' + str(Brakes[i]) + '^' +  "Car Velocity:" + '^' + str(car[i].velocity)  + '^' +  "Car acceleration:" + '^' + str(car[i].acceleration) + "\n")
    
    
            
            #Collision detection
            try:
                if (bg_sprite.get_at((int(car[i].position.x),int(car[i].position.y)))!=ROAD_COLOR):
                    car[i].alive = False
                    car[i].fitness -= time_since_start
            except:

                car[i].alive = False
                car[i].fitness -= time_since_start
            
            car_rect = pygame.Rect(car[i].position.x,car[i].position.y,10,10)
            index = car_rect.collidelist(Rect_list)

            if index != -1:

                if car[i].index == (index + 1)  % len(Rect_list):
                    #print("Going back", i, car[i].index, index)
                    #Exit_RS.write("Going back: " + '^' + str(i) + '^' + str(car[i].index) + '^' + str(index)+ "\n")
                    car[i].alive = False
                    car[i].fitness = -10000
                else:
                    #print("Going forward", i, car[i].index, index)
                    #Exit_RS.write("Going forward: " + '^' + str(i) + '^' + str(car[i].index) + '^' + str(index)+ "\n")
                    car[i].index = index
                    
            if time_since_start > 200 and car[i].velocity.y == 0:
                car[i].alive = False
                car[i].fitness -= time_since_start
       
            
            if time_since_start > 2000:
                car[i].alive = False
                car[i].fitness -= time_since_start
                
                
        alive.append(car[i].alive)
        fitness.append(car[i].fitness)
            
    first_car_index = [b for b, j in enumerate(fitness) if j == max(fitness)][0]
    
    if (sum(alive) == 1):
        #Showing fitness of the last alive
        index_last = [i for i, x in enumerate(alive) if x][0]
        fitness_info = myfont.render('Fitness of the last alive: ' + str(car[index_last].fitness) + ' Acceleration: ' + str(car[index_last].acceleration) + ' Velocity: ' + str(car[index_last].velocity) , False, (255, 255, 255))
        win.blit(fitness_info,(5,SCREEN_HEIGHT + 25))

        
    if (not any(alive)):
        
        # Crossover: the best two parents exchange their genes (weights)
        crossover(fitness)
        
        #Introducing mutations: with a probability of 5% each weight can be changed by a number between -0.1 to +0.1
        mutate()
        
        Generation += 1
        
        for i in range(0,NUMBER_MODELS):
            car[i].position = Vector2(CAR_INIT_X,CAR_INIT_Y)
            car[i].velocity = Vector2(0.0, -1*CAR_SPEED)
            car[i].angle = CAR_INIT_ANGLE
            car[i].fitness = 0
            car[i].index = 0
            car[i].alive = True
            time_since_start = 0
           
    

    alive = []
    fitness = []     
    
    #Showing generation
    win.blit(generation_info,(5,SCREEN_HEIGHT + 2))
    
    #for i in range(0,len(Rect_list)):
    #    pygame.draw.rect(win, (34, 139, 34), Rect_list[i])
        
    

    
    #pygame.display.update()
    pygame.display.flip()
    

pygame.quit()
