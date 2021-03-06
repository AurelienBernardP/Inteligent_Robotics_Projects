import numpy as np
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

    PALLET = np.array([[112, 128, 144],   # unexplored - grey
                       [255,   0,   0],   # obstacle - red
                       [  0, 255,   0],   # free - green
                       [  0,   0, 255],   # bot - blue
                       [255, 255,   0],   # ray - yellow
                       [255,   0, 255],   # frontier - pink
                       [255, 215,   0],   # Padding - orange
                       [0  ,   0,   0]])  # Route - black


    def __init__(self, width, height):

        self.map_size = (width,height) #(x_size,y_size) in matrix cells
        self.occupancy_matrix = np.zeros((height, width), dtype=int)

        self.real_room_size = (15,15) # (x_size,y_size) in meters
        
        self.bot_pos = np.zeros(2) #(x,y)in meters matrix position
        self.bot_orientation = 0.0 # in radians

        self.ray_endings = [] # matrix of Nb_rays columns and each column = (x,y,z)
        self.ray_hit = []

        self.frontier_cells = set()
        self.frontier_cells_list = []

        self.free_cells = set()
        self.obstacle_cells = set()
        self.padding_cells = set()
        self.explored_cells = set()
    
    def update_bot_pos(self, new_pos, new_orientation):
        self.bot_pos[0] = new_pos[0]
        self.bot_pos[1] = new_pos[1]
        self.bot_orientation = new_orientation
    '''
    def is_frontier(self,x,y):

        for i in range(x-1,x+2,1):
            for j in range(y-1,y+2,1):
                if(i < 0 or j < 0 or i >= np.shape(self.occupancy_matrix)[0] or j >= np.shape(self.occupancy_matrix)[1]):
                    continue
                elif(self.occupancy_matrix[i,j] == Scene_map.UNEXPLORED):
                    return True

        return False

    def add_padding(self,x,y):
        
        for i in range(x-2,x+3,1):
            for j in range(y-2,y+3,1):
                if(i < 0 or j < 0 or i >= np.shape(self.occupancy_matrix)[0] or j >= np.shape(self.occupancy_matrix)[1]):
                    continue
                
                elif(self.occupancy_matrix[i,j] == Scene_map.FREE):
                    self.occupancy_matrix[i,j] = Scene_map.PADDING

        return 
    '''
    def add_padding_v2(self,position):
        if position in self.obstacle_cells:
            # padding was already done
            return
        (x,y) = position
        for i in range(x-2,x+3,1):
            for j in range(y-2,y+3,1):
                if(i < 0 or j < 0 or i >= np.shape(self.occupancy_matrix)[0] or j >= np.shape(self.occupancy_matrix)[1]):
                    continue
                self.padding_cells.add((i,j))

    def is_frontier_v2(self,position):
        (x,y) = position
        for i in range(x-1,x+2,1):
            for j in range(y-1,y+2,1):
                if(i < 0 or j < 0 or i >= np.shape(self.occupancy_matrix)[0] or j >= np.shape(self.occupancy_matrix)[1]):
                    return True
                if((i,j) not in self.explored_cells and not (i,j) in self.frontier_cells and (i,j) != (x,y)):
                    return True
        return False

    def update_contact_map_v2(self,points,occupancy):
        self.ray_endings = np.hstack((points[:3,:],points[3:,:]))
        self.ray_hit = np.reshape(occupancy,(np.shape(occupancy)[0]*np.shape(occupancy)[1]))

        new_frontiers = set()
        c, s = np.cos(self.bot_orientation), np.sin(self.bot_orientation)
        R = np.array(((c, -s), (s, c)))

        for i in range(0,np.shape(self.ray_endings)[1],4):

            new_ray_coordinates = np.dot(R,(self.ray_endings[0,i],self.ray_endings[1,i]))

            #ray ending coordinate in metric
            self.ray_endings[0,i] = self.bot_pos[0] + new_ray_coordinates[0]
            self.ray_endings[1,i] = self.bot_pos[1] + new_ray_coordinates[1]

            #get the matrix position of the elements of interest
            ray_x,ray_y = self.map_position_to_mat_index(self.ray_endings[0,i],self.ray_endings[1,i])

            if(self.ray_hit[i]):
                self.add_padding_v2((ray_x,ray_y))
                self.obstacle_cells.add((ray_x,ray_y))
                self.explored_cells.add((ray_x,ray_y))
                self.free_cells.discard((ray_x,ray_y))

                if (ray_x,ray_y) in self.frontier_cells:
                    self.frontier_cells.remove((ray_x,ray_y))
                    self.frontier_cells_list.remove((ray_x,ray_y))

                
            elif((ray_x,ray_y) not in self.explored_cells):
                new_frontiers.add((ray_x,ray_y))

            bot_x,bot_y = self.map_position_to_mat_index(self.bot_pos[0],self.bot_pos[1])
            #update the state of all cells before the end of the ray
            ray_cells = line_generation(bot_x,bot_y, ray_x, ray_y)
            ray_cells.pop() # remove last cell which is the hit cell

            for cell in ray_cells:
                if(cell in self.explored_cells):
                    continue

                if(self.is_frontier_v2(cell)):
                    new_frontiers.add(cell)

                else:
                    self.free_cells.add(cell)
                    self.explored_cells.add(cell)
                    if(cell in self.frontier_cells):
                        self.frontier_cells.discard(cell)
                        self.frontier_cells_list.remove(cell)

                new_frontiers = new_frontiers.difference(self.frontier_cells)
                self.frontier_cells_list.extend(new_frontiers)
                self.frontier_cells = self.frontier_cells.union(new_frontiers)
                                  
        return
    '''
    def update_contact_map_from_sensor(self,points,occupancy):

        self.ray_endings = np.hstack((points[:3,:],points[3:,:]))
        self.ray_hit = np.reshape(occupancy,(np.shape(occupancy)[0]*np.shape(occupancy)[1]))

        bot_x,bot_y = self.map_position_to_mat_index(self.bot_pos[0],self.bot_pos[1])

        for i in range(0,np.shape(self.ray_endings)[1],3):

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
                    continue
                if(self.occupancy_matrix[cell[0],cell[1]] == Scene_map.FREE):
                    continue

                if(self.is_frontier(cell[0],cell[1])):
                    self.occupancy_matrix[cell[0],cell[1]] = Scene_map.FRONTIER
                    if (cell[0],cell[1]) not in self.frontier_cells :
                        self.frontier_cells.add(cell)
                        self.frontier_cells_list.append(cell)
                    
                elif self.occupancy_matrix[cell[0],cell[1]] != Scene_map.PADDING :
                    self.occupancy_matrix[cell[0],cell[1]] = Scene_map.FREE
                    if(cell in self.frontier_cells):
                        self.frontier_cells.discard(cell)
                        self.frontier_cells_list.remove(cell)
                                
    def pygame_screen_refresh(self, screen, init_pos_route, route):

        x_screen_size, y_screen_size = screen.get_size()
        circle_size = (np.minimum(x_screen_size,y_screen_size)/self.map_size[0])/2

        #background
        screen.fill((112, 128, 144))
        
        #draw occupancy map
        for i in range(np.shape(self.occupancy_matrix)[1]):
            for j in range(np.shape(self.occupancy_matrix)[0]):
                
                pygame.draw.circle(screen, Scene_map.PALLET[self.occupancy_matrix[i][j]], self.index_to_screen_position(x_screen_size,y_screen_size,j,i),circle_size )

        bot_x,bot_y = self.map_position_to_mat_index(self.bot_pos[0],self.bot_pos[1])

        
        #draw rays
        for i in range(0,np.shape(self.ray_endings)[1],3):
            pygame.draw.line(screen, Scene_map.PALLET[Scene_map.RAY],self.get_screen_pos_from_map_coordinates(screen,self.bot_pos), self.get_screen_pos_from_map_coordinates(screen,(self.ray_endings[0,i],self.ray_endings[1,i])))

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
                pygame.draw.line(screen,Scene_map.PALLET[Scene_map.ROUTE],self.index_to_screen_position(x_screen_size,y_screen_size,current_node[0],current_node[1]),self.index_to_screen_position(x_screen_size,y_screen_size,end_line_pos[0],end_line_pos[1]),width=int(2*circle_size))
                current_node = end_line_pos
        #Draw robot
        pygame.draw.circle(screen, Scene_map.PALLET[Scene_map.BOT], self.get_screen_pos_from_map_coordinates(screen,self.bot_pos),2*circle_size)
        
        c, s = np.cos(self.bot_orientation), np.sin(self.bot_orientation)
        R = np.array(((c, -s), (s, c)))
        line_end = np.dot(R,(0,-0.5))

        line_start = self.bot_pos
        line_end = np.add(line_end,self.bot_pos)

        line_start = self.get_screen_pos_from_map_coordinates(screen,self.bot_pos)
        line_end = self.get_screen_pos_from_map_coordinates(screen,line_end)
        pygame.draw.line(screen, Scene_map.PALLET[Scene_map.BOT], line_start, line_end, width = 3)

        '''
    def pygame_screen_refresh_v2(self, screen, init_pos_route, route):
        x_screen_size, y_screen_size = screen.get_size()
        circle_size = (np.minimum(x_screen_size,y_screen_size)/self.map_size[0])/2

        #background
        screen.fill((112, 128, 144))
        
        #draw cells
        for cell in self.free_cells:
            cell = (cell[1],cell[0])
            pygame.draw.circle(screen, Scene_map.PALLET[Scene_map.FREE], self.index_to_screen_position(x_screen_size,y_screen_size,cell[0],cell[1]),circle_size )

        for cell in self.frontier_cells:
            cell = (cell[1],cell[0])
            pygame.draw.circle(screen, Scene_map.PALLET[Scene_map.FRONTIER], self.index_to_screen_position(x_screen_size,y_screen_size,cell[0],cell[1]),circle_size )
        
        surface = pygame.Surface((x_screen_size,y_screen_size), pygame.SRCALPHA)
        for cell in self.padding_cells:
            cell = (cell[1],cell[0])
            pygame.draw.circle(surface, np.append(Scene_map.PALLET[Scene_map.PADDING],200), self.index_to_screen_position(x_screen_size,y_screen_size,cell[0],cell[1]),circle_size )
        screen.blit(surface, (0,0))
        
        for cell in self.obstacle_cells:
            cell = (cell[1],cell[0])
            pygame.draw.circle(screen, Scene_map.PALLET[Scene_map.OBSTACLE], self.index_to_screen_position(x_screen_size,y_screen_size,cell[0],cell[1]),circle_size )


        #draw rays
        for i in range(0,np.shape(self.ray_endings)[1],4):
            pygame.draw.line(screen, Scene_map.PALLET[Scene_map.RAY],self.get_screen_pos_from_map_coordinates(screen,self.bot_pos), self.get_screen_pos_from_map_coordinates(screen,(self.ray_endings[0,i],self.ray_endings[1,i])))

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
                pygame.draw.line(screen,Scene_map.PALLET[Scene_map.ROUTE],self.index_to_screen_position(x_screen_size,y_screen_size,current_node[0],current_node[1]),self.index_to_screen_position(x_screen_size,y_screen_size,end_line_pos[0],end_line_pos[1]),width=int(2*circle_size))
                current_node = end_line_pos
        #Draw robot
        pygame.draw.circle(screen, Scene_map.PALLET[Scene_map.BOT], self.get_screen_pos_from_map_coordinates(screen,self.bot_pos),2*circle_size)
        
        c, s = np.cos(self.bot_orientation), np.sin(self.bot_orientation)
        R = np.array(((c, -s), (s, c)))
        line_end = np.dot(R,(0,-0.5))

        line_start = self.bot_pos
        line_end = np.add(line_end,self.bot_pos)

        line_start = self.get_screen_pos_from_map_coordinates(screen,self.bot_pos)
        line_end = self.get_screen_pos_from_map_coordinates(screen,line_end)
        pygame.draw.line(screen, Scene_map.PALLET[Scene_map.BOT], line_start, line_end, width = 3)

        return
        
    def index_to_screen_position(self,screen_width,screen_height,x,y):

        cell_width = screen_width/self.map_size[0]
        cell_height = screen_height/self.map_size[1]
        #y axis is flipped so to get the y at the correct position we have to flip it again by doing y = y_size - pos
        return (int(x * cell_width),screen_height - int( y * cell_height))

    
    def map_position_to_mat_index(self,x,y):

        cells_per_meter_y = self.map_size[1] // self.real_room_size[1]
        cells_per_meter_x = self.map_size[0] // self.real_room_size[0]

        return np.minimum(int(y*cells_per_meter_y + self.map_size[1]/2), self.map_size[1]-1),np.minimum(int(x*cells_per_meter_x + self.map_size[0]/2),self.map_size[0]-1)


    def getCellType(self, cell):
        if 0 <= cell[0] <= self.map_size[0]-1 and 0 <= cell[1] <= self.map_size[1]-1:
            return self.occupancy_matrix[cell[0]][cell[1]]
        else:
            return -1
    
    
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

    def get_screen_pos_from_map_coordinates(self,screen,map_pos):
        x_screen_size, y_screen_size = screen.get_size()

        #screen coordinates start from top left and y axis is flipped
        #map coordinates start in the middle of the map and follow regular conventions

        screen_pos_x = (map_pos[0] + (self.real_room_size[0]/2)) * (x_screen_size / self.real_room_size[0])
        screen_pos_y = 700 - ((map_pos[1] + (self.real_room_size[1]/2)) * (y_screen_size/ self.real_room_size[1]))

        return (screen_pos_x,screen_pos_y)

def manhattanDistance(state_1, state_2):
    return abs(state_1[0] - state_2[0]) + abs(state_1[1] - state_2[1])


def line_generation(x0,y0,x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1) using Bresenham algorithm.
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