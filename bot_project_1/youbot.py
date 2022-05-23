# -*- coding: utf-8 -*-
"""
The aim of this code is to show small examples of controlling the displacement of the robot in V-REP. 
(C) Copyright Renaud Detry 2013, Mathieu Baijot 2017, Norman Marlier 2019.
Distributed under the GNU General Public License.
(See http://www.gnu.org/copyleft/gpl.html)
"""
# VREP
from multiprocessing.connection import wait
from Scene_map_v2 import manhattanDistance
import sim as vrep

# Useful import
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import pygame

from cleanup_vrep import cleanup_vrep
from vrchk import vrchk
from youbot_init import youbot_init
from youbot_drive import youbot_drive
from youbot_hokuyo_init import youbot_hokuyo_init
from youbot_hokuyo import youbot_hokuyo
from youbot_xyz_sensor import youbot_xyz_sensor
from beacon import beacon_init, youbot_beacon
from utils_sim import angdiff


from Scene_map_v2 import Scene_map
from astar import getActions
from astar import getAngle
from PID_controller import PID_controller

pygame.init()
screen = pygame.display.set_mode([700, 700])


# Test the python implementation of a youbot
# Initiate the connection to the simulator.
print('Program started')
# Use the following line if you had to recompile remoteApi
# vrep = remApi('remoteApi', 'extApi.h')
# vrep = remApi('remoteApi')

# Close the connection in case if a residual connection exists
vrep.simxFinish(-1)
clientID = vrep.simxStart('127.0.0.1',  19997, True, True, 2000, 5)

# The time step the simulator is using (your code should run close to it).
timestep = .05

# Synchronous mode
returnCode = vrep.simxSynchronous(clientID, True)

# If you get an error like:
#   Remote API function call returned with error code: 64.
# Explanation: simxStart was not yet called.
# Make sure your code is within a function!
# You cannot call V-REP from a script.
if clientID < 0:
    sys.exit('Failed connecting to remote API server. Exiting.')

print('Connection ' + str(clientID) + ' to remote API server open')

# Make sure we close the connection whenever the script is interrupted.
# cleanup_vrep(vrep, id)

# This will only work in "continuous remote API server service".
# See http://www.v-rep.eu/helpFiles/en/remoteApiServerSide.htm
vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

# Send a Trigger to the simulator: this will run a time step for the physics engine
# because of the synchronous mode. Run several iterations to stabilize the simulation

for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)


# Retrieve all handles, mostly the Hokuyo.
h = youbot_init(vrep, clientID)
h = youbot_hokuyo_init(vrep, h)
beacons_handle = beacon_init(vrep, clientID, h)


# Send a Trigger to the simulator: this will run a time step for the physics engine
# because of the synchronous mode. Run several iterations to stabilize the simulation
for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)

# Time
t_run = []

##############################################################################
#                                                                            #
#                          INITIAL CONDITIONS                                #
#                                                                            #
##############################################################################
# Define all the variables which will be used through the whole simulation.
# Important: Set their initial values.

# Get the position of the beacons in the world coordinate frame (x, y)
# simx_opmode_oneshot_wait is used. This enforces to have a valid response.
beacons_world_pos = np.zeros((len(beacons_handle), 3))
for i, beacon in enumerate(beacons_handle):   
    res, beacons_world_pos[i] = vrep.simxGetObjectPosition(clientID, beacon, -1,
                                                           vrep.simx_opmode_oneshot_wait)

# Parameters for controlling the youBot's wheels: at each iteration,
# those values will be set for the wheels.
# They are adapted at each iteration by the code.
forwBackVel = 0  # Move straight ahead.
rightVel = 0  # Go sideways.
rotateRightVel = 0  # Rotate.

# First state of state machine
fsm = 'rotate'
print('Switching to state: ', fsm)

# Get the initial position
res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
# Set the speed of the wheels to 0.
h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

# Send a Trigger to the simulator: this will run a time step for the physic engine
# because of the synchronous mode. Run several iterations to stabilize the simulation.
for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)


house_map = Scene_map(75,75)

# Actions that will come from A* algo.
actions = [('Est', 0.001)]
currActionIndex = 0
goalCell = (-1,-1)

# To track position at the beginning of a move.
youbotFirstPos = youbotPos

def get_speeds(map_rep,real_bot_position,target_pos_mat,current_orientation,target_orientation):
    
    goal_coordinates = map_rep.get_cell_center_coordinates(target_pos_mat[1],target_pos_mat[0])

    map_dir_vector = np.subtract(goal_coordinates,real_bot_position) # (x,y) vector
    euler_dist = np.linalg.norm(map_dir_vector,2)
    map_dir_vector = map_dir_vector / euler_dist

    theta = current_orientation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    bot_dir_vector = np.dot(R,map_dir_vector)

    if target_orientation == None :
        target_orientation = np.arctan(map_dir_vector[0]/map_dir_vector[1])
    right_speed = - 2 * euler_dist * bot_dir_vector[0]
    up_speed =  2 * euler_dist * bot_dir_vector[1]
    rot_speed = 2 * angdiff(current_orientation, target_orientation)

    return (right_speed,up_speed,rot_speed)



# Start the demo. 
intial_pos_route = (0,0)
counter = 0
show = True
forward_PID = PID_controller(timestep,3,0.8,0,True)
rot_PID = PID_controller(timestep,3.05,0.8,0,True)

while True:
    try:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                exit()

        # Time management
        t_loop = time.perf_counter()

        # Check the connection with the simulator
        if vrep.simxGetConnectionId(clientID) == -1:
            sys.exit('Lost connection to remote API.')

        # Get the position and the orientation of the robot.
        res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_streaming)
        vrchk(vrep, res, True) # Check the return value from the previous V-REP call (res) and exit in case of error.
        res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_streaming)
        vrchk(vrep, res, True)

        # Get youbot state.
        state = house_map.map_position_to_mat_index(youbotPos[0], youbotPos[1])

        house_map.update_bot_pos((youbotPos[0],youbotPos[1]),youbotEuler[2])

        # Get the distance from the beacons
        # Change the flag to True to constraint the range of the beacons
        beacon_dist = youbot_beacon(vrep, clientID, beacons_handle, h, flag=False)

        # Get data from the hokuyo - return empty if data is not captured
        scanned_points, contacts = youbot_hokuyo(vrep, h, vrep.simx_opmode_buffer)
        vrchk(vrep, res)
       

        if counter % 5 == 0 or counter < 5:
            #update map and refresh display every 5 ticks except at the begining where a lot of data is gathered
            house_map.update_contact_map(scanned_points,contacts)
            show = True
        if show == 1:
            house_map.pygame_screen_refresh(screen,intial_pos_route,actions)
            pygame.display.flip()        
            show = False
        
        
        if counter == 550:
            forward_PID.plot()
            rot_PID.plot()
        
        #print(counter,end='\r')
        counter +=1
        
        # Apply the state machine.
        if fsm == 'planning':

            currActionIndex = 0

            house_map.update_contact_map(scanned_points,contacts) # to remove

            # Set the goal state.
            cellNextToGoal = (-1,-1)
            while cellNextToGoal == (-1,-1):
                goalCell = house_map.frontier_cells_list[len(house_map.frontier_cells_list)-1] # take the newest frontier point

                # Turn the goal state to be the fist free cell next to the goal cell.
                for i in range(goalCell[0]-1,goalCell[0]+2,1):
                        for j in range(goalCell[1]-1,goalCell[1]+2,1):
                            if (i,j) in house_map.free_cells and (i,j) not in house_map.padding_cells:
                                cellNextToGoal = (i,j)
                if cellNextToGoal == (-1,-1):
                    house_map.frontier_cells.discard(goalCell)
                    house_map.frontier_cells_list.remove(goalCell)

            # Set actions to take.
            actions = getActions(house_map, cellNextToGoal)
            print("actions", actions)

            if len(actions) == 0:
                fsm = 'planning'
                print('Switching to state: ', fsm)
            else:
                fsm = 'rotate'
                print('Switching to state: ', fsm)
                intial_pos_route = (youbotPos[0],youbotPos[1])

        elif fsm == 'rotate':
            # Compute the value of the left and right angles.
            angle1 = youbotEuler[2]
            angle2 = getAngle(actions[currActionIndex][0])
            if (angle1 >= angle2):
                angleRight = 2 * np.pi - (np.pi - angle1) - (np.pi + angle2)
            else:
                angleRight = (np.pi - angle2) + (np.pi + angle1)
            angleLeft = 2 * np.pi - angleRight
            
            # Rotate left or right (choose the best of the two move).
            if (angleRight <= angleLeft):
                distanceToGoal = angleRight
                rotateRightVel = - 1/3 * distanceToGoal
            else:
                distanceToGoal = -angleLeft
                rotateRightVel = 1/3 * distanceToGoal

            rotateRightVel = rot_PID.control(0,distanceToGoal)

            # Stop when the robot reached the goal angle.
            if abs(distanceToGoal) < .01 and abs(rotateRightVel) < 0.1:
                rotateRightVel = 0
                fsm = 'moveFoward'
                print('Switching to state: ', fsm)

        
        elif fsm == 'moveFoward':
            
            # Compute the distance already travelled for this move.
            currActionType = actions[currActionIndex][0]
            if (currActionType == 'Est' or currActionType == 'West'):
                distance = abs(youbotPos[0] - youbotFirstPos[0])
            else:
                distance = abs(youbotPos[1] - youbotFirstPos[1])

            # Compute the distance that remain to travel.
            distanceToGoal = actions[currActionIndex][1] - distance

            # Set the speed to reach the goal.
            forwBackVel = forward_PID.control(0,distanceToGoal)
            #forwBackVel = - 0.5 * distanceToGoal  # to remove

            # Stop when the robot reached the goal position.
            if abs(distanceToGoal) < .01 and abs(forwBackVel) < 0.1:
                forwBackVel = 0  # Stop the robot.

                # Perform the next action or do planning if no action remain.
                currActionIndex = currActionIndex + 1
                youbotFirstPos = youbotPos
                if (currActionIndex >= len(actions)):
                    fsm = 'stop'
                    print('Switching to state: ', fsm)
                else:
                    fsm = 'rotate'
                    print('Switching to state: ', fsm)
            
            # Stop if we explored the goal cell and we are close.
            elif goalCell not in house_map.frontier_cells and manhattanDistance(goalCell, state) < 15:
                fsm = 'stop'
                print('Switching to state: ', fsm)
        

        elif fsm == 'stop':
            forwBackVel = 0  # Stop the robot.
            
            # Check if the youbot is stoped.
            if abs(youbotPos[0] - youbotFirstPos[0]) + abs(youbotPos[1] - youbotFirstPos[1]) <= .01:
                fsm = 'planning'
                print('Switching to state: ', fsm)
            
            youbotFirstPos = youbotPos

            if len(house_map.frontier_cells) == 0:
                fsm = 'finished'
                print('Switching to state: ', fsm)


        elif fsm == 'finished':
            print('Finished exploration, no more accessible cells in frontier')
            time.sleep(3)
            break


        else:
            sys.exit('Unknown state ' + fsm)

        # Update wheel velocities.
        h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

        # What happens if you do not update the velocities?
        # The simulator always considers the last speed you gave it,
        # until you set a new velocity.

        # Send a Trigger to the simulator: this will run a time step for the physic engine
        # because of the synchronous mode.
        end_time = time.perf_counter()
        t_run.append((end_time-t_loop)*1000.)  # In ms
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)
    except KeyboardInterrupt:
        cleanup_vrep(vrep, clientID)
        sys.exit('Stop simulation')

cleanup_vrep(vrep, clientID)
print('Simulation has stopped')
# Histogram of time loop
n, x, _ = plt.hist(t_run, bins=100)
plt.vlines(np.min(t_run), 0, np.max(n), linewidth=1.5, colors="r")
plt.vlines(np.max(t_run), 0, np.max(n), linewidth=1.5, colors="k")
plt.xlabel(r"time $t_{\rm{loop}}$ (ms)")
plt.ylabel("Number of loops (-)")
plt.show()