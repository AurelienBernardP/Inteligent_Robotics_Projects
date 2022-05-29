# -*- coding: utf-8 -*-
"""
The aim of this code is to show small examples of controlling the displacement of the robot in V-REP. 
(C) Copyright Renaud Detry 2013, Mathieu Baijot 2017, Norman Marlier 2019.
Distributed under the GNU General Public License.
(See http://www.gnu.org/copyleft/gpl.html)
"""
# VREP
from multiprocessing.connection import wait
from turtle import xcor
from Scene_map_v3 import manhattanDistance
import sim as vrep

# Useful import
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import pygame
import math 

from cleanup_vrep import cleanup_vrep
from vrchk import vrchk
from youbot_init import youbot_init
from youbot_drive import youbot_drive
from youbot_hokuyo_init import youbot_hokuyo_init
from youbot_hokuyo import youbot_hokuyo
from youbot_xyz_sensor import youbot_xyz_sensor
from beacon import beacon_init, youbot_beacon
from utils_sim import angdiff
from scipy.spatial.transform import Rotation as R

import cv2 as cv

from Scene_map_v3 import Scene_map
from astar import getActions
from astar import getAngle
from astar import getRigthLeftAngles
from PID_controller import PID_controller
#from youbot_arm import get_transform

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
rot_counter = 0
fsm = 'main'
print('Switching to state: ', fsm)

# Get the initial position
res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
# Set the speed of the wheels to 0.
h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)
# Get the target orientation
[res, targetori] = vrep.simxGetObjectOrientation(clientID, h["otarget"], h["r22"], vrep.simx_opmode_oneshot_wait)
# Get the gripper orientation
[res, tori] = vrep.simxGetObjectOrientation(clientID, h["otip"], h["r22"], vrep.simx_opmode_oneshot_wait)

# Send a Trigger to the simulator: this will run a time step for the physic engine
# because of the synchronous mode. Run several iterations to stabilize the simulation.
for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)


house_map = Scene_map(75,75,beacons_world_pos[:,:2])

# Actions that will come from A* algo.
actions = [('East', 0.001)]
currActionIndex = 0
goalCell = (-1,-1)

fsm_step = 1

tableCenter = [0,0]

# To track position at the beginning of a move.
youbotFirstPos = youbotPos

# Put the fct elsewhere !

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


def getRotationSpeed(angle1, angle2):

    angleRight, angleLeft = getRigthLeftAngles(angle1, angle2)
            
    # Rotate left or right (choose the best of the two move).
    if (angleRight <= angleLeft):
        distanceToGoal = angleRight
        rotateRightVel = - 1/3 * distanceToGoal
    else:
        distanceToGoal = -angleLeft
        rotateRightVel = 1/3 * distanceToGoal

    rotateRightVel = rot_PID.control(0,distanceToGoal)

    return rotateRightVel, distanceToGoal


def setArmJoints(targetJoint):

    # Joint 0.
    res = vrep.simxSetJointTargetPosition(clientID, h["armJoints"][0], targetJoint[0], vrep.simx_opmode_oneshot)            
    res, joint_0 = vrep.simxGetJointPosition(clientID, h["armJoints"][0], vrep.simx_opmode_buffer)

    # Joint 1.   
    res = vrep.simxSetJointTargetPosition(clientID, h["armJoints"][1], targetJoint[1], vrep.simx_opmode_oneshot)            
    res, joint_1 = vrep.simxGetJointPosition(clientID, h["armJoints"][1], vrep.simx_opmode_buffer)

    # Joint 3.
    res = vrep.simxSetJointTargetPosition(clientID, h["armJoints"][3], targetJoint[3], vrep.simx_opmode_oneshot)            
    res, joint_3 = vrep.simxGetJointPosition(clientID, h["armJoints"][3], vrep.simx_opmode_buffer)

    return joint_0, joint_1, joint_3



# Start the demo. 
intial_pos_route = (0,0)
counter = 0
show = False
forward_PID = PID_controller(timestep,3,0.8,0,True)
rot_PID = PID_controller(timestep,3.05,0.8,0,True)
tables = np.zeros((3,3))
known_table1 = np.asarray([-3,-6])
known_table2 = np.asarray([-1,-6])
target_table = np.zeros(2)
target_table = [-1, -6] # to remove
youbotFirstEuler = -1
gripperState = 1

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
        #res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_streaming)
        #vrchk(vrep, res, True) # Check the return value from the previous V-REP call (res) and exit in case of error.
        res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_streaming)
        vrchk(vrep, res, True)       

        # Get the distance from the beacons
        # Change the flag to True to constraint the range of the beacons
        beacon_dist = youbot_beacon(vrep, clientID, beacons_handle, h, flag=False)
        house_map.update_bot_pos(beacon_dist,youbotEuler[2])
        # Get data from the hokuyo - return empty if data is not captured
        scanned_points, contacts = youbot_hokuyo(vrep, h, vrep.simx_opmode_buffer)
        vrchk(vrep, res)
        

        # Get th youbot position given by the triangulation instead of gps (milestone 1.ii).
        youbotPos = house_map.bot_pos_estimate

        # Get youbot state.
        state = house_map.map_position_to_mat_index(youbotPos[0], youbotPos[1])
       

        if show == True:
            house_map.pygame_screen_refresh(screen,intial_pos_route,actions,beacon_dist)
            for i in range(np.shape(tables)[0]):
                x_screen_size, y_screen_size = screen.get_size()
                x = (tables[i,0] + (15 / 2)) * (x_screen_size/15)
                y = y_screen_size - ((tables[i,1] + (15 / 2)) * (y_screen_size/15))
                radius = tables[i,2] * (x_screen_size/15)
                pygame.draw.circle(screen, (255, 0, 0), (x,y), radius)
            pygame.display.flip()        
            show = False
        if counter % 5 == 0 or counter < 5:
            #update map and refresh display every 5 ticks except at the begining where a lot of data is gathered
            house_map.update_contact_map(scanned_points,contacts)
            show = True

        
        
        #if counter == 550:
            #forward_PID.plot()
            #rot_PID.plot()
        
        #print(counter,end='\r')
        counter +=1
        if fsm == 'detect_tables' or (counter % 250 == 0):

            img = np.copy(house_map.obstacle_cells_grid)
            img = (img + 1) % 2
            img *= 255
            img = np.uint8(img)
            img = cv.blur(img, (3, 3))
            circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 7,
                            param1=50, param2=10,
                            minRadius=8, maxRadius=9)
            if circles is not None:
                circles = np.uint16(np.around(circles))

                for i,pt in enumerate(circles[0, :]):
                    a, b, r = pt[0], pt[1], pt[2]
                    tables[i,0] = (a - (0.5*300)) * (15/300)
                    tables[i,1] = (b - (0.5*300)) * (15/300)
                    tables[i,2] = r * (15/300) 

                    d1 = np.linalg.norm(known_table1.T - tables[i,:2])
                    d2 = np.linalg.norm(known_table2.T - tables[i,:2])

                    if np.linalg.norm(known_table1.T - tables[i,:2]) > 0.8 and np.linalg.norm(known_table2.T - tables[i,:2]) > 0.8 :
                        print('found target table')
                        target_table[0] = tables[i,0]
                        target_table[1] = tables[i,1]
                    print(target_table)
            else :
                print("no tables detected")


        # Apply the state machine.
        if fsm == 'main':

            # Explore the whole map.
            if fsm_step == 1:
                fsm = 'exploring'
                print('Switching to state: ', fsm)
            
            # Move objects from tables to tables.
            if fsm_step == 2:
                fsm = 'moveObject'
                print('Switching to state: ', fsm)

        
        elif fsm == 'moveObject':

                # test (to replace by point cloud processing)
                angleOfObject = np.pi
                centerOfObject = np.array([-0.02, 0.37, 0.26])
                
                # Infinit loop (test)
                print(tableCenter)
                print(known_table1)
                if (tableCenter == known_table1).all():
                    tableCenter = target_table
                else:
                    tableCenter = known_table1

                tableCenterCell = house_map.map_position_to_mat_index(tableCenter[0],tableCenter[1])
                
                # Find a cell close to the table and the youbot.
                closestFreeCells = set(x for x in house_map.free_cells if (manhattanDistance(x, tableCenterCell) <= 8 and x not in house_map.padding_cells))
                def f(x):
                    return manhattanDistance(x, state)
                cellNextToGoal = min(closestFreeCells,key=f)

                print(cellNextToGoal)

                # Set actions to take.
                currActionIndex = 0
                actions = getActions(house_map, cellNextToGoal)
                print("actions", actions)
                
                if len(actions) == 0:
                    fsm = 'moveObject'
                    print('Switching to state: ', fsm)
                else:
                    youbotFirstPos = youbotPos
                    print(state)
                    fsm = 'rotate'
                    print('Switching to state: ', fsm)
                    intial_pos_route = (youbotPos[0],youbotPos[1])


        elif fsm == 'exploring':

            currActionIndex = 0

            house_map.update_contact_map(scanned_points,contacts) # to remove ? 

            # Check if the map is fully explored.
            isMapFullyExplored = house_map.frontier_cells_list == []
            
            # Set the goal state (if map not fully explored).
            if not isMapFullyExplored:
                cellNextToGoal = (-1,-1)
                while cellNextToGoal == (-1,-1) and  len(house_map.frontier_cells):
                    goalCell = house_map.frontier_cells_list[-1] # take the newest frontier point

                    # Turn the goal state to be the fist free cell next to the goal cell.
                    for i in range(goalCell[0]-1,goalCell[0]+2,1):
                            for j in range(goalCell[1]-1,goalCell[1]+2,1):
                                if (i,j) in house_map.free_cells and (i,j) not in house_map.padding_cells:
                                    cellNextToGoal = (i,j)
                    if cellNextToGoal == (-1,-1):
                        house_map.frontier_cells.discard(goalCell)
                        house_map.frontier_cells_list.remove(goalCell)
            
            # Check if the map is fully explored.
            isMapFullyExplored = house_map.frontier_cells_list == []
            isMapFullyExplored = True #to remove
            
            # Set actions to take.
            actions = getActions(house_map, cellNextToGoal)
            print("actions", actions)

            if isMapFullyExplored:
                fsm_step = 2
                fsm = 'main'
                print('Switching to state: ', fsm)
            elif len(actions) == 0:
                house_map.frontier_cells.discard(goalCell)
                house_map.frontier_cells_list.remove(goalCell)
                fsm = 'exploring'
                print('Switching to state: ', fsm)
            elif len(actions) != 0:
                youbotFirstPos = youbotPos
                fsm = 'rotate'
                print('Switching to state: ', fsm)
                intial_pos_route = (youbotPos[0],youbotPos[1])


        elif fsm == 'rotate':
            # Compute the value of the left and right angles.
            angle1 = youbotEuler[2]
            angle2 = getAngle(actions[currActionIndex][0])

            rotateRightVel, distanceToGoal = getRotationSpeed(angle1, angle2)

            # Stop when the robot reached the goal angle.
            if abs(distanceToGoal) < .01 and abs(rotateRightVel) < 0.1:
                rotateRightVel = 0
                fsm = 'moveFoward'
                print('Switching to state: ', fsm)

        
        elif fsm == 'moveFoward':
            
            # Compute the distance already travelled for this move.
            distance = math.sqrt(abs(youbotPos[0] - youbotFirstPos[0])**2 + abs(youbotPos[1] - youbotFirstPos[1])**2)

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
            elif goalCell not in house_map.frontier_cells and manhattanDistance(goalCell, state) < 10 and fsm_step == 1:
                fsm = 'stop'
                print('Switching to state: ', fsm)

        
        elif fsm == 'rotateToTable':
 
            dx = tableCenter[0] - youbotPos[0]
            dy = tableCenter[1] - youbotPos[1]

            if dx >= 0 and dy >= 0:
                angle = np.pi/2 + abs(math.atan(dy/dx))
            if dx >= 0 and dy <= 0:
                angle = abs(math.atan(dx/dy))
            if dx <= 0 and dy >= 0:
                angle = -(np.pi/2 + abs(math.atan(dy/dx)))
            if dx <= 0 and dy <= 0:
                angle = -abs(math.atan(dx/dy))

            angle1 = youbotEuler[2] 
            angle2 = angle

            rotateRightVel, distanceToGoal = getRotationSpeed(angle1, angle2)
            
            # Stop when the robot reached the goal angle.
            if abs(distanceToGoal) < .01 and abs(rotateRightVel) < 0.1:
                rotateRightVel = 0
                fsm = 'moveToTable'
                print('Switching to state: ', fsm)
        

        elif fsm == 'moveToTable':

            distanceToCenter = math.sqrt(abs(youbotPos[0] - tableCenter[0])**2 + abs(youbotPos[1] - tableCenter[1])**2) # use triangulation !
            distanceToGoal = distanceToCenter - 0.85

            # Set the speed to reach the goal.
            forwBackVel = forward_PID.control(0,distanceToGoal) * 0.2

            # Stop when the robot reached the goal position.
            if abs(distanceToGoal) < .01 and abs(forwBackVel) < 0.1:
                forwBackVel = 0
                fsm = 'circleAroundTable'
                print('Switching to state: ', fsm)


        elif fsm == 'circleAroundTable':
            
            forwBackVel = 0
            
            # We need to be at distance 0.850 m from table center and face it !
            
            # Set the goal angle.
            angle2 = angleOfObject

            # Get "table angle" from youbot angle.
            angle1 = youbotEuler[2]
            if angle1 <= 0:
                angle1 = np.pi + angle1
            else:
                angle1 = angle1 - np.pi

            # Compute the value of the left and right angles.
            angleRight, angleLeft = getRigthLeftAngles(angle1, angle2)
            
            # Rotate left or right (choose the best of the two move).
            if (angleRight <= angleLeft):
                rotateRightVel = -0.09
                rightVel = 0.1
            else:
                rotateRightVel = 0.09
                rightVel = -0.1
            
            # Stop when the robot reached the goal angle.
            distanceToGoal = abs(angle1 - angle2)
            if distanceToGoal < .01:
                rotateRightVel = 0
                rightVel = 0
                fsm = 'halfTurn'
                print('Switching to state: ', fsm)
            

        elif fsm == 'halfTurn':

            #need to set "arg" youbotFirstEuler first !
            if youbotFirstEuler == -1:
                youbotFirstEuler = youbotEuler

            angle1 = youbotEuler[2]

            if youbotFirstEuler[2] < 0:
                angle2 = youbotFirstEuler[2] + np.pi
            else:
                angle2 = youbotFirstEuler[2] - np.pi

            rotateRightVel, distanceToGoal = getRotationSpeed(angle1, angle2)
            
            # Stop when the robot reached the goal angle.
            if abs(distanceToGoal) < .01 and abs(rotateRightVel) < 0.1:
                rotateRightVel = 0
                fsm = 'deployArm'
                print('Switching to state: ', fsm)

        
        elif fsm == 'deployArm':

            targetJoint = [np.pi, -np.pi/4, 0., 0.]
            
            # Rotate the arm.
            joint_0, joint_1, joint_3 = setArmJoints(targetJoint)

            # Stop when the robot is at an angle close to target.
            cond0 = abs(angdiff(joint_0, targetJoint[0])) < .001
            cond1 = abs(angdiff(joint_1, targetJoint[1])) < .001
            cond3 = abs(angdiff(joint_3, targetJoint[3])) < .001
            
            if cond0 & cond1 & cond3:
                res = vrep.simxSetIntegerSignal(clientID, 'km_mode', 2, vrep.simx_opmode_oneshot_wait)
                fsm = 'moveArm'
                print('Switching to state: ', fsm)

            
        elif fsm == 'moveArm':

            # Send command to the robot arm.
            res = vrep.simxSetObjectPosition(clientID, h["ptarget"], h["armRef"], centerOfObject, vrep.simx_opmode_oneshot)
            vrchk(vrep, res, True)

            # Get the gripper position and check whether it is at destination (the original position).
            [res, tpos] = vrep.simxGetObjectPosition(clientID, h["ptip"], h["armRef"], vrep.simx_opmode_buffer)
            vrchk(vrep, res, True)

            # Check only position but orientation can be added.
            cond_pos = np.linalg.norm(tpos - centerOfObject) < .005
            print(np.linalg.norm(tpos - centerOfObject))

            # Close the gripper if we must grab an object and open it if we must drop it.
            if cond_pos:
                fsm = 'activateGripper'
                print('Switching to state: ', fsm)
                time_to_close = time.time()

        
        elif fsm == 'activateGripper':
            
            # Open or close the gripper.
            if gripperState == 1:
                close = 0
            else:
                close = 1
            res = vrep.simxSetIntegerSignal(clientID, 'gripper_open', close, vrep.simx_opmode_oneshot_wait);
            #vrchk(vrep, res)
            
            if time.time()-time_to_close > 3.:
                gripperState = close
                fsm = 'liftUp'
                res = vrep.simxSetIntegerSignal(clientID, 'km_mode', 0, vrep.simx_opmode_oneshot_wait)
                #vrchk(vrep, res, True)
                print('Switching to state: ', fsm)

                
        elif fsm == "liftUp":

            # Joint 3
            target_joint_3 = -np.pi/2
            joint_index = 3
            res = vrep.simxSetJointTargetPosition(clientID, h["armJoints"][joint_index], target_joint_3, vrep.simx_opmode_oneshot)            
            res, joint_3 = vrep.simxGetJointPosition(clientID, h["armJoints"][joint_index], vrep.simx_opmode_buffer)

            # Condition
            cond = abs(angdiff(joint_3, target_joint_3)) < .001
            if cond:
                fsm = "storeArm"
                print('Switching to state: ', fsm)

        
       # Need to be dynamic
        elif fsm == 'storeArm':

            targetJoint = [0, 0, 0., 0]
            
            # Rotate the arm.
            joint_0, joint_1, joint_3 = setArmJoints(targetJoint)

            # Stop when the robot is at an angle close to target.
            cond0 = abs(angdiff(joint_0, targetJoint[0])) < .001
            cond1 = abs(angdiff(joint_1, targetJoint[1])) < .001
            cond3 = abs(angdiff(joint_3, targetJoint[3])) < .001
            
            if cond0 & cond1 & cond3:
                fsm = 'main'
                print('Switching to state: ', fsm)
        




    
        elif fsm == 'stop':
            forwBackVel = 0  # Stop the robot.
            
            # Check if the youbot is stoped.
            if abs(youbotPos[0] - youbotFirstPos[0]) + abs(youbotPos[1] - youbotFirstPos[1]) <= .01:
                if fsm_step == 1:
                    fsm = 'exploring'
                    print('Switching to state: ', fsm)
                if fsm_step == 2:
                    fsm = 'rotateToTable'
                    print('Switching to state: ', fsm)
            
            youbotFirstPos = youbotPos


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