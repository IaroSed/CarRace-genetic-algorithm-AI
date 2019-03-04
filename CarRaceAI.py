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
    classifier[i].add(Dense(output_dim = 4, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
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
    best_classifiers[i].add(Dense(output_dim = 4, kernel_initializer='random_uniform', activation = 'relu', input_dim = 5))
    # Adding the output layer
    best_classifiers[i].add(Dense(output_dim = 2, kernel_initializer='random_uniform', activation = 'sigmoid'))
    
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

#Exit_RS = open("ResultsSelection.txt", 'w',encoding='utf-8')

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
        
        self.alive = True
        
        self.fitness = 0
        
        self.index = 0

    
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
        
        
        #Updating the fitness
        self.fitness += CAR_SPEED 
        



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
    #pygame.draw.circle(win, (255,0,0), (int(xf), int(yf)), 2)
    
    # Right stream
    for i in range(0, SCREEN_WIDTH):
        xr = int(position.x + i *cos(2*pi + radians(-a)))
        yr = int(position.y + i *sin(2*pi + radians(-a)))
        if (bg_sprite.get_at((xr,yr))!=ROAD_COLOR):
            break;
            
    dr = sqrt((position.x-xr)**2 + (position.y-yr)**2)        
    #pygame.draw.circle(win, (255,0,0), (int(xr), int(yr)), 2)
    
    # Left stream
    for i in range(0, SCREEN_WIDTH):
        xl = int(position.x + i *cos(pi + radians(-a)))
        yl = int(position.y + i *sin(pi + radians(-a)))
        if (bg_sprite.get_at((xl,yl))!=ROAD_COLOR):
            break;
    
    dl = sqrt((position.x-xl)**2 + (position.y-yl)**2)        
    #pygame.draw.circle(win, (255,0,0), (int(xl), int(yl)), 2)
    
    # Right 45deg stream
    for i in range(0, SCREEN_WIDTH):
        xdr = int(position.x + i *cos(1.75*pi + radians(-a)))
        ydr = int(position.y + i *sin(1.75*pi + radians(-a)))
        if (bg_sprite.get_at((xdr,ydr))!=ROAD_COLOR):
            break;
    
    ddr = sqrt((position.x-xdr)**2 + (position.y-ydr)**2)        
    #pygame.draw.circle(win, (255,0,0), (int(xdr), int(ydr)), 2)
    
    # Left 45deg stream
    for i in range(0, SCREEN_WIDTH):
        xdl = int(position.x + i *cos(1.25*pi + radians(-a)))
        ydl = int(position.y + i *sin(1.25*pi + radians(-a)))
        if (bg_sprite.get_at((xdl,ydl))!=ROAD_COLOR):
            break;
    
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
        
        A = classifier[i].get_weights()[0][0]
        B = classifier[i].get_weights()[0][1]
        C = classifier[i].get_weights()[0][2]
        D = classifier[i].get_weights()[0][3]
        E = classifier[i].get_weights()[0][4]
        Z = classifier[i].get_weights()[2]
    
        #number_changes = 0
        
        for j in range(0,3):
            
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                A[j] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C),list(D),list(E)]) , np.zeros(4, dtype=float), Z , np.zeros(2, dtype=float)])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                B[j] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C),list(D),list(E)]) , np.zeros(4, dtype=float), Z , np.zeros(2, dtype=float)])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                C[j] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C),list(D),list(E)]) , np.zeros(4, dtype=float), Z , np.zeros(2, dtype=float)])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                D[j] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C),list(D),list(E)]) , np.zeros(4, dtype=float), Z , np.zeros(2, dtype=float)])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                E[j] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C),list(D),list(E)]) , np.zeros(4, dtype=float), Z , np.zeros(2, dtype=float)])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                Z[j][0] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C),list(D),list(E)]) , np.zeros(4, dtype=float), Z , np.zeros(2, dtype=float)])
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                Z[j][1] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C),list(D),list(E)]) , np.zeros(4, dtype=float), Z , np.zeros(2, dtype=float)])
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


run = True

while run:
    
    pygame.time.delay(5)
    
    generation_info = myfont.render('Generation ' + str(Generation) + " Best fitness: " + str(best_fitness[0]) , False, (255, 255, 255))
    
    
    dt = 1

    win.fill((0,0,0))
    
    win.blit(bg_sprite, (0,0))

    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            
            run = False
    
    for i in range(0,NUMBER_MODELS):        
        
        #keys = pygame.key.get_pressed()
        
        rotated = pygame.transform.rotate(car_image, car[i].angle)
        car_rect = rotated.get_rect()
        
        win.blit(rotated, Vector2(car[i].position)  - (car_rect.width / 2, car_rect.height / 2))
        
               
        car[i].steering = max(-car[i].max_steering, min(car[i].steering, car[i].max_steering))

        
        if (car[i].alive == True):
            car[i].update(dt)
        
        # Calculating the distances around the car - inputs of the ANN model
        X[i].iloc[0,0], X[i].iloc[0,1],X[i].iloc[0,2],X[i].iloc[0,3],X[i].iloc[0,4] = calculate_distances(car[i].position,car[i].angle)
 
       
        # Predicting the Test set results:
        Turn_right[i] = classifier[i].predict(X[i])[0][0]
        Turn_left[i] = classifier[i].predict(X[i])[0][1]
        
        Turn_right[i] = (Turn_right[i] > 0.5)
        Turn_left[i] = (Turn_left[i] > 0.5)
    
        if Turn_left[i]:
            car[i].steering += 30 * dt
        elif Turn_right[i]:
            car[i].steering -= 30 * dt
        else:
            car[i].steering = 0

        
        #Collision detection
        if (bg_sprite.get_at((int(car[i].position.x),int(car[i].position.y)))!=ROAD_COLOR):
            car[i].alive = False
        
        car_rect = pygame.Rect(car[i].position.x,car[i].position.y,10,10)
        index = car_rect.collidelist(Rect_list)
        #print(index)
        if index != -1:
           # print("Collision")
            if car[i].index == (index + 1)  % len(Rect_list):
                #print("Going back", i, car[i].index, index)
                #Exit_RS.write("Going back: " + '^' + str(i) + '^' + str(car[i].index) + '^' + str(index)+ "\n")
                car[i].alive = False
                car[i].fitness = -100
            else:
                #print("Going forward", i, car[i].index, index)
                #Exit_RS.write("Going forward: " + '^' + str(i) + '^' + str(car[i].index) + '^' + str(index)+ "\n")
                car[i].index = index
   
        alive.append(car[i].alive)
        fitness.append(car[i].fitness)
    
    if (sum(alive) == 1):
        #Showing fitness of the last alive
        fitness_info = myfont.render('Fitness of the last alive: ' + str(car[[i for i, x in enumerate(alive) if x][0]].fitness) , False, (255, 255, 255))
        win.blit(fitness_info,(5,SCREEN_HEIGHT + 25))
        #Score
        #score_info = myfont.render('Score: ' + str(car[[i for i, x in enumerate(alive) if x][0]].score) , False, (0, 0, 0))
        #win.blit(score_info,(5, 5))    
        
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
           
    
    alive = []
    fitness = []     
    
    #Showing generation
    win.blit(generation_info,(5,SCREEN_HEIGHT + 2))
    
    #for i in range(0,len(Rect_list)):
    #    pygame.draw.rect(win, (34, 139, 34), Rect_list[i])
        
    

    
    #pygame.display.update()
    pygame.display.flip()
    

pygame.quit()