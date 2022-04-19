# -*- coding: utf-8 -*-
"""
The aim of this code is to show small examples of controlling the displacement of the robot in V-REP. 
(C) Copyright Renaud Detry 2013, Mathieu Baijot 2017, Norman Marlier 2019.
Distributed under the GNU General Public License.
(See http://www.gnu.org/copyleft/gpl.html)
"""
# VREP
from multiprocessing.connection import wait
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


from Scene_map import Scene_map
from astar import getActions

pygame.init()
screen = pygame.display.set_mode([700, 700])
screen.fill((255, 255, 255))

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


house_map = Scene_map(150,150)

# Actions that will come from A* algo.
goalCell = (0,0)
actions = [('Sud', 0)]
currActionIndex = 0

# To track position at the beginning of a move.
youbotFirstPos = youbotPos

# Define a function to get the angle corresponding to each move.
def getAngle(x):
    return {
            'North': -np.pi,
            'Sud': 0,
            'Est': np.pi/2,
            'West': -np.pi/2,
     }[x]


# Start the demo. 
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

        house_map.update_bot_pos((youbotPos[0],youbotPos[1]),youbotEuler[2])

        # Get the distance from the beacons
        # Change the flag to True to constraint the range of the beacons
        beacon_dist = youbot_beacon(vrep, clientID, beacons_handle, h, flag=False)

        # Get data from the hokuyo - return empty if data is not captured
        scanned_points, contacts = youbot_hokuyo(vrep, h, vrep.simx_opmode_buffer)
        vrchk(vrep, res)
       
        
        # is it to slow ? dont work with my part.
        #start = time.time()
        
        house_map.update_contact_map_from_sensor(scanned_points,contacts)
        house_map.pygame_screen_refresh(screen)
        pygame.display.flip()
    
        #end = time.time()
        #total_time = end - start
        #print("\n"+ str(total_time))
       
     

        # Apply the state machine.
        if fsm == 'planning':

            currActionIndex = 0
            
            # Set goal position
            goalCell = (-1,-1)
            for i in range(0,150,1):
                if goalCell != (-1,-1):
                    break
                for j in range(0,150,1):
                    if (house_map.getCellType((i,j)) == house_map.FRONTIER):
                        goalCell = (i,j)
                        print(goalCell)
                        actions = getActions(house_map, goalCell) # set goal state here instead of in A*
                        print(actions)
                        if len(actions) == 0:
                            goalCell = (-1,-1)
                        else:
                            break


            if len(actions) == 0:
                fsm = 'planning'
                print('Switching to state: ', fsm)
            else:
                fsm = 'rotate'
                print('Switching to state: ', fsm)

        
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
                distanceToGoal = angleLeft
                rotateRightVel = 1/3 * distanceToGoal
            
            # Stop when the robot reached the goal angle.
            if distanceToGoal < .002:
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
            forwBackVel = - 0.5 * distanceToGoal
            
            # Stop when the robot reached the goal position.
            if abs(distanceToGoal) < .01:
                forwBackVel = 0  # Stop the robot.

                # Perform the next action or do planning if no action remain.
                currActionIndex = currActionIndex + 1
                youbotFirstPos = youbotPos
                if (currActionIndex >= len(actions)):
                    fsm = 'planning'
                    print('Switching to state: ', fsm)
                else:
                    fsm = 'rotate'
                    print('Switching to state: ', fsm)

            # Stop if we explored the goal cell.
            elif house_map.getCellType(goalCell) != house_map.FRONTIER and distanceToGoal < 2 and currActionIndex == len(actions)-1:
                fsm = 'stop'
                print('Switching to state: ', fsm)
        

        elif fsm == 'stop':
            forwBackVel = 0  # Stop the robot.
            
            # Check if the youbot is stoped.
            if abs(youbotPos[0] - youbotFirstPos[0]) + abs(youbotPos[1] - youbotFirstPos[1]) <= .01:
                fsm = 'planning'
                print('Switching to state: ', fsm)
            
            youbotFirstPos = youbotPos


        elif fsm == 'finished':
            print('Finish')
            time.sleep(3)
            break


        else:
            sys.exit('Unknown state ' + fsm)

            
            





        '''
        # To remove -----------------------------
        elif fsm == 'forward':

            # Make the robot drive with a constant speed (very simple controller, likely to overshoot). 
            # The speed is - 1 m/s, the sign indicating the direction to follow. Please note that the robot has
            # limitations and cannot reach an infinite speed. 
            forwBackVel = -1

            # Stop when the robot is close to y = - 6.5. The tolerance has been determined by experiments: if it is too
            # small, the condition will never be met (the robot position is updated every 50 ms); if it is too large,
            # then the robot is not close enough to the position (which may be a problem if it has to pick an object,
            # for example). 
            if abs(youbotPos[1] + 6.5) < .02:
                forwBackVel = 0  # Stop the robot.
                fsm = 'backward'
                print('Switching to state: ', fsm)


        elif fsm == 'backward':
            # A speed which is a function of the distance to the destination can also be used. This is useful to avoid
            # overshooting: with this controller, the speed decreases when the robot approaches the goal. 
            # Here, the goal is to reach y = -4.5. 
            forwBackVel = - 2 * (youbotPos[1] + 4.5)
            # distance to goal influences the maximum speed

            # Stop when the robot is close to y = 4.5.
            if abs(youbotPos[1] + 4.5) < .01:
                forwBackVel = 0  # Stop the robot.
                fsm = 'right'
                print('Switching to state: ', fsm)

        elif fsm == 'right':
            # Move sideways, again with a proportional controller (goal: x = - 4.5). 
            rightVel = - 2 * (youbotPos[0] + 4.5)

            # Stop at x = - 4.5
            if abs(youbotPos[0] + 4.5) < .01:
                rightVel = 0  # Stop the robot.
                fsm = 'rotateRight'
                print('Switching to state: ', fsm)

        elif fsm == 'rotateRight':
            # Rotate until the robot has an angle of -pi/2 (measured with respect to the world's reference frame). 
            # Again, use a proportional controller. In case of overshoot, the angle difference will change sign, 
            # and the robot will correctly find its way back (e.g.: the angular speed is positive, the robot overshoots, 
            # the anguler speed becomes negative). 
            # youbotEuler(3) is the rotation around the vertical axis.              
            rotateRightVel = angdiff(youbotEuler[2], (np.pi/2))

            # Stop when the robot is at an angle close to -pi/2.
            if abs(angdiff(youbotEuler[2], (-np.pi/2))) < .002:
                rotateRightVel = 0
                fsm = 'finished'
                print('Switching to state: ', fsm)
        '''

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