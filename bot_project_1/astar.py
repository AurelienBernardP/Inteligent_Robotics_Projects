## implementation inspired from IA course project implementation
## add credit !

import numpy as np
from queue import PriorityQueue
import random

from Scene_map import Scene_map
from Scene_map import *


def getActions(houseMap):
    
    actions = []

    path = __astar__(houseMap)

    #print(path)

    if (len(path) == 0):
        return actions

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

    #print(actions)

    return actions


def __astar__(houseMap):

    # Set the variables.
    path = []
    pathCost = 0
    closed = set()

    # Get youbot initial state.
    youbotPos = houseMap.bot_pos
    state = map_position_to_mat_index(youbotPos[0], youbotPos[1])

    # Set goal position
    goalCell = random.choice(houseMap.frontier_cells)
    
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
            childPathCost = pathCost + 1 # create g(state) function instead ?
            priority = childPathCost + __heuristic__(nextState, goalCell)
            fringe.put((priority, (nextState, path + [action], childPathCost)))


# To heavy ?
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
    return __manhattanDistance__(state, goalState)


def __manhattanDistance__(state_1, state_2):
    return abs(state_1[0] - state_2[0]) + abs(state_1[1] - state_2[1])
            

def __generateYoubotSuccessors__(state, houseMap):

    youbotSuccessors = []
    
    # Generate the state resulting from moving north.
    nextState = (state[0]+1, state[1])
    nextStateType = getCellType(houseMap, nextState)
    if (nextStateType != houseMap.OBSTACLE and nextStateType != houseMap.UNEXPLORED):
        youbotSuccessors.append((nextState, 'North'))
    
    # Generate the state resulting from moving sud.
    nextState = (state[0]-1, state[1])
    nextStateType = getCellType(houseMap, nextState)
    if (nextStateType != houseMap.OBSTACLE and nextStateType != houseMap.UNEXPLORED):
        youbotSuccessors.append((nextState, 'Sud'))
    
    # Generate the state resulting from moving west.
    nextState = (state[0], state[1]-1)
    nextStateType = getCellType(houseMap, nextState)
    if (nextStateType != houseMap.OBSTACLE and nextStateType != houseMap.UNEXPLORED):
        youbotSuccessors.append((nextState, 'West'))
    
    # Generate the state resulting from moving est.
    nextState = (state[0], state[1]+1)
    nextStateType = getCellType(houseMap, nextState)
    if (nextStateType != houseMap.OBSTACLE and nextStateType != houseMap.UNEXPLORED):
        youbotSuccessors.append((nextState, 'Est'))

    return youbotSuccessors



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
houseMap.frontier_cells.append((10,53))

getActions(houseMap)
'''