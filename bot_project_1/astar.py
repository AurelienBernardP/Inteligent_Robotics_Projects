## implementation inspired from IA course project implementation
## add credit !

import numpy as np
from queue import PriorityQueue
import random

from Scene_map import Scene_map
from Scene_map import *


def getActions(houseMap, goalCell):
    
    actions = []

    path = __astar__(houseMap, goalCell)

    if (len(path) == 0):
        return actions

    # Convert the actions to a format readable by the youbot.
    cellDistance = 15 / houseMap.map_size[0] # to modify (hard coded)
    distanceToTravel = cellDistance
    currAction = path[0]
    for i in range(len(path) - 1):
        if path[i+1] == currAction:
            distanceToTravel += cellDistance
        else:
            actions.append((currAction, round(distanceToTravel, 2)))
            distanceToTravel = cellDistance
            currAction = path[i+1]
    actions.append((currAction, round(distanceToTravel, 2)))

    return actions


def __astar__(houseMap, goalCell):

    # Set the variables.
    path = []
    pathCost = 0
    closed = set()

    # Get youbot initial state.
    youbotPos = houseMap.bot_pos
    state = houseMap.map_position_to_mat_index(youbotPos[0], youbotPos[1])

    print(state)
    
    # Set the fringe.
    fringe = PriorityQueue()
    fringe.put((0, (state, path, pathCost)))

    # Apply the Astar algorithm and return the list of actions.
    while True:
        if fringe.empty():
            return []
    
        # Pop the fringe.
        priority, (current, path, pathCost) = fringe.get()

        # Check if we reach the goal.
        if current == goalCell:
            return path

        # Check for cycle.
        if current in closed:
            continue
        closed.add(current)

        # Generate the children and add it to the fringe.
        for nextState, action in __generateYoubotSuccessors__(current, houseMap):
            # Push the current child in the fringe.
            childPathCost = pathCost + __costFunction__(nextState, action, path, houseMap)
            priority = childPathCost + __heuristic__(nextState, goalCell)
            fringe.put((priority, (nextState, path + [action], childPathCost)))


# To heavy ? not so good
'''
def __heuristic__(state, houseMap):

    distanceToFrontier = houseMap.map_size[0] + houseMap.map_size[1]
    for frontierState in houseMap.frontier_cells:
        distanceToCell = __manhattanDistance__(state, frontierState)
        if distanceToCell < distanceToFrontier:
            distanceToFrontier = distanceToCell
    
    return distanceToFrontier
'''

def __heuristic__(state, goalState):
    return manhattanDistance(state, goalState)


def __costFunction__(nextState, action, path, houseMap):
    cost = 0
    cost += 1 # time
    if len(path) != 0 and action != path[len(path)-1]:
        cost += 5 + 50 * 1 / len(path)
    if houseMap.getCellType(nextState) == houseMap.PADDING:
        cost += 100000

    return cost
    

            

def __generateYoubotSuccessors__(state, houseMap):

    youbotSuccessors = []
    
    # Generate the state resulting from moving north.
    nextState = (state[0]+1, state[1])
    nextStateType = houseMap.getCellType(nextState)
    if (nextStateType != houseMap.OBSTACLE and nextStateType != houseMap.UNEXPLORED and nextStateType != -1):
        youbotSuccessors.append((nextState, 'North'))
    
    # Generate the state resulting from moving sud.
    nextState = (state[0]-1, state[1])
    nextStateType = houseMap.getCellType(nextState)
    if (nextStateType != houseMap.OBSTACLE and nextStateType != houseMap.UNEXPLORED and nextStateType != -1):
        youbotSuccessors.append((nextState, 'Sud'))
    
    # Generate the state resulting from moving west.
    nextState = (state[0], state[1]-1)
    nextStateType = houseMap.getCellType(nextState)
    if (nextStateType != houseMap.OBSTACLE and nextStateType != houseMap.UNEXPLORED and nextStateType != -1):
        youbotSuccessors.append((nextState, 'West'))
    
    # Generate the state resulting from moving est.
    nextState = (state[0], state[1]+1)
    nextStateType = houseMap.getCellType(nextState)
    if (nextStateType != houseMap.OBSTACLE and nextStateType != houseMap.UNEXPLORED and nextStateType != -1):
        youbotSuccessors.append((nextState, 'Est'))

    return youbotSuccessors


# to test

'''
path = ['North', 'North', 'North', 'North', 'Sud', 'Sud', 'Sud', 'Est', 'Est', 'Est', 'Est', 'North', 'Est', 'Sud', 'Sud']

actions = []

cellDistance = 15 / 150 # to modify (hard coded)

distanceToTravel = cellDistance
currAction = path[0]
for i in range(len(path) - 1):
    if path[i+1] == currAction:
        distanceToTravel += cellDistance
    else:
        actions.append((currAction, round(distanceToTravel, 2)))
        distanceToTravel = cellDistance
        currAction = path[i+1]
actions.append((currAction, round(distanceToTravel, 2)))




print(actions)
'''

'''
houseMap = Scene_map(150,150)
houseMap.bot_pos = (-2.12, -4.97)
for i in range (35):
    for j in range (100):
        houseMap.occupancy_matrix[i][j] = 2

houseMap.occupancy_matrix[10][53] = 5

getActions(houseMap)
'''



'''
state = (31,3)
list = [(32,45), (43,3), (21,23)]

goalCell =  min(list,key = lambda x: __manhattanDistance__(x,state)) 
print(goalCell)
'''