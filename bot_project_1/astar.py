## implementation inspired from my last year IA course project implementation
## add credit !

from queue import PriorityQueue

import numpy as np
import math

from Scene_map_v2 import manhattanDistance

def getActions(houseMap, goalCell):

    path = __astar__(houseMap, goalCell)

    if (len(path) == 0):
        return []

    # Compute the distance beetwen two adjacent cells in the map.
    cellDistance = houseMap.real_room_size[0] / houseMap.map_size[0]

    actions = __convertPathToActions__(cellDistance, path)

    return actions


def __astar__(houseMap, goalCell):

    # Set the variables.
    path = []
    pathCost = 0
    closed = set()

    # Get youbot initial state.
    youbotPos = houseMap.bot_pos_estimate
    state = houseMap.map_position_to_mat_index(youbotPos[0], youbotPos[1])

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


def __heuristic__(state, goalState):
    #return manhattanDistance(state, goalState)
    return math.sqrt((state[0] - goalState[0])**2 + (state[1] - goalState[1])**2) # or manatthan ?


def __costFunction__(state, action, path, houseMap):
    cost = 0
    cost += 1 # time

    if isMoveDiagonal(action):
        cost += math.sqrt(2) - 1

    if len(path) != 0 and action != path[len(path)-1]:
        cost += 500
    if state in houseMap.padding_cells:
        cost += 100000
    
    # Penalized an initial rotation of the youbot.
    if len(path) == 0:
        actionAngle = getAngle(action)
        youbotAngle = houseMap.bot_orientation
        if abs(actionAngle - youbotAngle) > 0.2:
            cost += 500

    return cost
    

def __generateYoubotSuccessors__(state, houseMap):

    youbotCandidateSucessors = []
    
    # Generate the state resulting from moving north.
    nextState = (state[0]+1, state[1])
    youbotCandidateSucessors.append((nextState, 'North'))

    # Generate the state resulting from moving north-east.
    nextState = (state[0]+1, state[1]+1)
    youbotCandidateSucessors.append((nextState, 'North-East'))

    # Generate the state resulting from moving east.
    nextState = (state[0], state[1]+1)
    youbotCandidateSucessors.append((nextState, 'East'))

    # Generate the state resulting from moving south-east.
    nextState = (state[0]-1, state[1]+1)
    youbotCandidateSucessors.append((nextState, 'South-East'))
    
    # Generate the state resulting from moving south.
    nextState = (state[0]-1, state[1])
    youbotCandidateSucessors.append((nextState, 'South'))

    # Generate the state resulting from moving south-west.
    nextState = (state[0]-1, state[1]-1)
    youbotCandidateSucessors.append((nextState, 'South-West'))
    
    # Generate the state resulting from moving west.
    nextState = (state[0], state[1]-1)
    youbotCandidateSucessors.append((nextState, 'West'))

    # Generate the state resulting from moving north-west.
    nextState = (state[0]+1, state[1]-1)
    youbotCandidateSucessors.append((nextState, 'North-West'))
    
    # Return the successor states that are valid.
    youbotSuccessors = []
    for youbotSuccessor in youbotCandidateSucessors:
        nextState = youbotSuccessor[0]
        nextStateType = houseMap.getCellType(nextState)
        if (nextState not in houseMap.obstacle_cells and nextState in houseMap.explored_cells and nextStateType != -1):
            youbotSuccessors.append(youbotSuccessor)

    return youbotSuccessors


def __convertPathToActions__(cellDistance, path):
    """Convert the path to a list of actions readable by the youbot.
       ...
    """

    actions = []
    
    # Convert the path to a list of actions.
    i = 0
    while i < len(path):
        # Fetch the current action type.
        currAction = path[i]

        # Set the distance to travel for the first action of the move.
        if isMoveDiagonal(path[i]):
            distanceToTravel = math.sqrt(cellDistance**2 + cellDistance**2)
        else:
            distanceToTravel = cellDistance
        
        # Increase the distance to travel if still same move.
        while i+1 < len(path) and currAction == path[i+1]:
            # Compute the current distance to travel.
            if isMoveDiagonal(currAction):
                currDistanceToTravel = math.sqrt(cellDistance**2 + cellDistance**2)
            else:
                currDistanceToTravel = cellDistance
        
            # Add the distance.
            distanceToTravel += currDistanceToTravel
            i = i + 1
        
        # Go the the next move.
        actions.append((currAction, round(distanceToTravel, 2)))
        i = i + 1

    return actions


# Define a function to get the angle corresponding to each move.
def getAngle(x):
    return {
            'North': -np.pi,
            'North-East': 3*np.pi/4,
            'East': np.pi/2,
            'South-East': np.pi/4,
            'South': 0,
            'South-West': -np.pi/4,
            'West': -np.pi/2,
            'North-West': -3*np.pi/4,
     }[x]

def isMoveDiagonal(action):
    if action in {'North-East', 'South-East', 'South-West', 'North-West'}:
        return True 
    else:
        return False



'''
print(isMoveDiagonal('South-East'))


path = ['North', 'East', 'North', 'West', 'West', 'West', 'West', 'West', 'South-East', 'South-East', 'South-East', 'South', 'South', 'North']


print(__convertPathToActions__(0.1, path))
'''

