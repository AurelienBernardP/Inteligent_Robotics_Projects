import numpy as np
import matplotlib.pyplot as plt
import pygame
import math

class Scene_map :

    UNEXPLORED = 0
    OBSTACLE = 1
    FREE = 2
    BOT = 3
    RAY = 4
    FRONTIER = 5
    PADDING = 6
    ROUTE = 7

    PALLET = np.array([[  255,   255,   255],   # unexplored - white
                    [255,   0,   0],   # obstacle - red
                    [  0, 255,   0],   # free - green
                    [  0,   0, 255],   # bot - blue
                    [255, 255,   0],   # ray - yellow
                    [255,   0, 255],   # frontier - pink
                    [255, 215,   0],   # Padding - orange
                    [0  ,   0,   0]])  # Route - black


    def __init__(self, width, height):

        self.map_size = (width,height)
        self.occupancy_matrix = np.zeros((height, width), dtype=int)
        self.real_room_size = (15,15) # in meters
        
        self.bot_pos = np.zeros(2) #(x,y)
        self.bot_orientation = 0.0
        self.ray_endings = [] # matrix of Nb_rays columns and each column = (x,y,z)
        self.ray_hit = []
        self.frontier_cells = set()


        '''
        Code to use if math plot is preferred to pygame

        plt.ion()

        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        RGB_map = Scene_map.PALLET[self.occupancy_matrix]
        self.line1 = self.ax.imshow(RGB_map)
        '''
        
        
    
    def update_bot_pos(self, new_pos, new_orientation):
        self.bot_pos[0] = new_pos[0]
        self.bot_pos[1] = new_pos[1]
        self.bot_orientation = new_orientation

    def is_frontier(self,x,y):

        for i in range(x-1,x+2,1):
            for j in range(y-1,y+2,1):
                if(i < 0 or j < 0 or i >= np.shape(self.occupancy_matrix)[0] or j >= np.shape(self.occupancy_matrix)[1]):
                    continue
                
                elif(self.occupancy_matrix[i,j] == Scene_map.UNEXPLORED):
                    self.frontier_cells.add((x,y))
                    return True

        return False

    def add_padding(self,x,y):
        
        for i in range(x-4,x+5,1):
            for j in range(y-4,y+5,1):
                if(i < 0 or j < 0 or i >= np.shape(self.occupancy_matrix)[0] or j >= np.shape(self.occupancy_matrix)[1]):
                    continue
                
                elif(self.occupancy_matrix[i,j] == Scene_map.FREE):
                    self.occupancy_matrix[i,j] = Scene_map.PADDING

        return 

    def update_contact_map_from_sensor(self,points,occupancy):

        self.ray_endings = np.hstack((points[:3,:],points[3:,:]))
        self.ray_hit = np.reshape(occupancy,(np.shape(occupancy)[0]*np.shape(occupancy)[1]))

        bot_x,bot_y = self.map_position_to_mat_index(self.bot_pos[0],self.bot_pos[1])

        for i in range(np.shape(self.ray_endings)[1]):

            #get ray coordinates in absolute value relative to robot
            theta = self.bot_orientation
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            new_ray_coordinates = np.dot(R,(self.ray_endings[0,i],self.ray_endings[1,i]))

            #ray ending coordinate
            self.ray_endings[0,i] = self.bot_pos[0] + new_ray_coordinates[0]
            self.ray_endings[1,i] = self.bot_pos[1] + new_ray_coordinates[1]

            #get the matrix position of the elements of interest
            ray_x,ray_y = self.map_position_to_mat_index(self.ray_endings[0,i],self.ray_endings[1,i])
            
            #does the ray hit an obstacle
            if(self.ray_hit[i] ):
                self.occupancy_matrix[ray_x,ray_y] = Scene_map.OBSTACLE
                self.add_padding(ray_x,ray_y)
            #update the state of all cells before the end of the ray
            ray_cells = line_generation(bot_x,bot_y, ray_x, ray_y)

            for cell in ray_cells:
                if(self.occupancy_matrix[cell[0],cell[1]] == Scene_map.OBSTACLE):
                    break
                if(self.occupancy_matrix[cell[0],cell[1]] == Scene_map.FREE):
                    continue

                if(self.is_frontier(cell[0],cell[1])):
                    self.occupancy_matrix[cell[0],cell[1]] = Scene_map.FRONTIER
                elif self.occupancy_matrix[cell[0],cell[1]] != Scene_map.PADDING :
                    self.occupancy_matrix[cell[0],cell[1]] = Scene_map.FREE
                    self.frontier_cells.discard(cell)




            

    def show_map_state(self):
        '''
        Code to use if math plot is preferred to pygame to update the screen
        '''

        #Real occupancy matrix as background
        drawn_matrix = np.copy(self.occupancy_matrix)

        #Overlay the robot and the covered area by the rays
        
        bot_x,bot_y = self.map_position_to_mat_index(self.bot_pos[0],self.bot_pos[1])

        #TODO : draw the rays 
        
        for i in range(np.shape(self.ray_endings)[1]):
            ray_x,ray_y = self.map_position_to_mat_index(self.ray_endings[0,i],self.ray_endings[1,i])
            irradiated_cells = line_generation(bot_x,bot_y, ray_x, ray_y)

            for j in range(len(irradiated_cells)-1):
                if(self.occupancy_matrix[irradiated_cells[j][0],irradiated_cells[j][1]] != Scene_map.OBSTACLE):
                    drawn_matrix[irradiated_cells[j][0],irradiated_cells[j][1]] = Scene_map.RAY
                else:
                    break
        
        #draw the bot       
        drawn_matrix[bot_x,bot_y] = Scene_map.BOT
        #cast drawn matrix to the RGB maping and update display
        RGB_map = Scene_map.PALLET[drawn_matrix]
        self.line1.set_data(RGB_map)

        plt.xlim(0,np.shape(self.occupancy_matrix)[0])
        plt.ylim(0,np.shape(self.occupancy_matrix)[1])

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def pygame_screen_refresh(self, screen, init_pos_route, route):

        x_screen_size, y_screen_size = screen.get_size()
        circle_size = np.minimum(x_screen_size,y_screen_size)/self.map_size[0]

        #draw occupancy map
        for i in range(np.shape(self.occupancy_matrix)[1]):
            for j in range(np.shape(self.occupancy_matrix)[0]):
                
                pygame.draw.circle(screen, Scene_map.PALLET[self.occupancy_matrix[i][j]], self.index_to_screen_position(x_screen_size,y_screen_size,j,i),circle_size )

        bot_x,bot_y = self.map_position_to_mat_index(self.bot_pos[0],self.bot_pos[1])

        #draw rays
        for i in range(np.shape(self.ray_endings)[1]):
            ray_x,ray_y = self.map_position_to_mat_index(self.ray_endings[0,i],self.ray_endings[1,i])
            pygame.draw.line(screen, Scene_map.PALLET[Scene_map.RAY], self.index_to_screen_position(x_screen_size,y_screen_size,bot_y,bot_x), self.index_to_screen_position(x_screen_size,y_screen_size,ray_y,ray_x))

        # draw route
            current_node = self.map_position_to_mat_index(init_pos_route[1],init_pos_route[0])
            cells_per_meter_y = self.map_size[1] / self.real_room_size[1]
            cells_per_meter_x = self.map_size[0] / self.real_room_size[0]
            for action in route:
                end_line_pos = (0,0)
                
                if action [0] =='Sud':
                    end_line_pos = (current_node[0], current_node[1] - action[1] * cells_per_meter_y)
                if action [0] =='North':
                    end_line_pos = (current_node[0], current_node[1] + action[1] * cells_per_meter_y)
                if action [0] =='Est':
                    end_line_pos = (current_node[0] + action[1] * cells_per_meter_x, current_node[1] )
                if action [0] =='West':
                    end_line_pos = (current_node[0] - action[1] * cells_per_meter_x, current_node[1] )
                pygame.draw.line(screen,Scene_map.PALLET[Scene_map.ROUTE],self.index_to_screen_position(x_screen_size,y_screen_size,current_node[0],current_node[1]),self.index_to_screen_position(x_screen_size,y_screen_size,end_line_pos[0],end_line_pos[1]),width=int(circle_size))
                current_node = end_line_pos
        #Draw robot
        pygame.draw.circle(screen, Scene_map.PALLET[Scene_map.BOT], self.index_to_screen_position(x_screen_size,y_screen_size,bot_y,bot_x),2*circle_size)
        
        
        c, s = np.cos(self.bot_orientation), np.sin(self.bot_orientation)
        R = np.array(((c, -s), (s, c)))
        line_end = np.dot(R,(0,-0.5))

        line_start = self.bot_pos
        line_end = np.add(line_end,self.bot_pos)
        line_start_mat_x,line_start_mat_y = self.map_position_to_mat_index(line_start[0],line_start[1])
        line_end_mat_x,line_end_mat_y = self.map_position_to_mat_index(line_end[0],line_end[1])

        line_start = self.index_to_screen_position(x_screen_size,y_screen_size,line_start_mat_y,line_start_mat_x)
        line_end = self.index_to_screen_position(x_screen_size,y_screen_size,line_end_mat_y,line_end_mat_x)
        pygame.draw.line(screen, Scene_map.PALLET[Scene_map.BOT], line_start, line_end, width = int(circle_size))

        
    def index_to_screen_position(self,screen_width,screen_height,x,y):

        cell_width = screen_width/self.map_size[0]
        cell_height = screen_height/self.map_size[1]
        #y axis is flipped so to get the y at the correct position we have to flip it again by doing y = y_size - pos
        return (int(x * cell_width),screen_height - int( y * cell_height))

    
    def map_position_to_mat_index(self,x,y):
        return np.minimum(math.floor(y*10 + 75), self.map_size[1]-1),np.minimum(math.floor(x*10 + 75),self.map_size[0]-1)


    def getCellType(self, cell):
        return self.occupancy_matrix[cell[0]][cell[1]]
    
    
    def get_cell_center_coordinates(self,x,y):

        cell_real_width  =  self.real_room_size[0] / self.map_size[0]# in meters
        cell_real_height =  self.real_room_size[1] / self.map_size[1]# in meters
        
        # Multiply by cell number and add half a cell in each direction to get the center of the cell
        x_coord = x * cell_real_width + cell_real_width /2
        y_coord = y * cell_real_height + cell_real_height / 2

        # Adjust for the fact that the matrix starts at the bottom left and real coordinates are centered in the middle of the map
        x_coord -= (self.real_room_size[0]/2)
        y_coord -= (self.real_room_size[1]/2)
        return ( x_coord , y_coord)


    
    # Not useful because we use padding.
    '''
    def isYoubotCollide(self, state):

        if state == self.UNEXPLORED:
            return True

        margin = 4 # to ajust
        for i in range(state[0]-margin,state[0]+margin+1,1):
            for j in range(state[1]-margin,state[1]+margin+1,1):
                if 0 <= i <= 149 and 0 <= j <= 149:
                    currCellType = self.getCellType((i,j))
                    if (currCellType == self.OBSTACLE):
                        return True
                else:
                    return True
                    
        return False
    '''


def manhattanDistance(state_1, state_2):
    return abs(state_1[0] - state_2[0]) + abs(state_1[1] - state_2[1])


def line_generation(x0,y0,x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.

    Credit : function code inspired by https://github.com/encukou/bresenham/blob/master/bresenham.py
    """

    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0
    cells = []
    for x in range(dx + 1):
        cells.append((x0 + x*xx + y*yx, y0 + x*xy + y*yy))
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy
         
    return cells