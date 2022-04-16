from cmath import tanh
import numpy as np
import matplotlib.pyplot as plt


class Scene_map :

    UNEXPLORED = 0
    OBSTACLE = 1
    FREE = 2
    BOT = 3
    RAY = 4
    FRONTIER = 5
    
    PALLET = np.array([[  255,   255,   255],   # unexplored - white
                    [255,   0,   0],   # obstacle - red
                    [  0, 255,   0],   # free - green
                    [  0,   0, 255],   # bot - blue
                    [255, 255,   0],   # ray - yellow
                    [255,   0, 255]])  # frontier - pink

    def __init__(self, width, height):
        self.map_size = (width,height)

        self.occupancy_matrix = np.zeros((height, width), dtype=int)
        
        self.bot_pos = np.zeros(2) #(x,y)
        self.bot_orientation = 0.0
        self.ray_endings = [] # matrix of Nb_rays columns and each column = (x,y,z)
        self.ray_hit = []
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        RGB_map = Scene_map.PALLET[self.occupancy_matrix]
        self.line1 = self.ax.imshow(RGB_map)
        
        
    
    def update_bot_pos(self, new_pos, new_orientation):
        self.bot_pos[0] = new_pos[0]
        self.bot_pos[1] = new_pos[1]
        self.bot_orientation = new_orientation

    def is_frontier(self,x,y):

        for i in range(x-1,x+2,1):
            for j in range(y-1,y+2,1):
                if(i< 0 or j<0 or i > np.shape(self.occupancy_matrix)[0] or i > np.shape(self.occupancy_matrix)[1]):
                    continue
                
                if(self.occupancy_matrix[i,j] == Scene_map.UNEXPLORED):
                    return True

        return False

    def update_contact_map_from_sensor(self,points,occupancy):

        self.ray_endings = np.hstack((points[:3,:],points[3:,:]))
        self.ray_hit = np.reshape(occupancy,(np.shape(occupancy)[0]*np.shape(occupancy)[1]))

        bot_x,bot_y = map_position_to_mat_index(self.bot_pos[0],self.bot_pos[1])

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
            ray_x,ray_y = map_position_to_mat_index(self.ray_endings[0,i],self.ray_endings[1,i])
            
            #does the ray hit an obstacle or a frontier?
            if(self.ray_hit[i]):
                self.occupancy_matrix[ray_x,ray_y] = Scene_map.OBSTACLE
            

            #update the state of all cells before the end of the ray
            ray_cells = line_generation(bot_x,bot_y, ray_x, ray_y)

            for j in range(len(ray_cells)):
                if(self.occupancy_matrix[ray_cells[j][0],ray_cells[j][1]] != Scene_map.OBSTACLE):
                    if(self.is_frontier(ray_cells[j][0],ray_cells[j][1])):
                        self.occupancy_matrix[ray_cells[j][0],ray_cells[j][1]] = Scene_map.FRONTIER
                    else:
                        self.occupancy_matrix[ray_cells[j][0],ray_cells[j][1]] = Scene_map.FREE
                else:
                    break
            

    def show_map_state(self):

        #Real occupancy matrix as background
        drawn_matrix = np.copy(self.occupancy_matrix)

        #Overlay the robot and the covered area by the rays
        
        bot_x,bot_y = map_position_to_mat_index(self.bot_pos[0],self.bot_pos[1])

        #TODO : draw the rays 
        
        for i in range(np.shape(self.ray_endings)[1]):
            ray_x,ray_y = map_position_to_mat_index(self.ray_endings[0,i],self.ray_endings[1,i])
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


def map_position_to_mat_index(x,y):
    return int(y*10 + 75),int(x*10 + 75)


def line_generation(x0,y0,x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
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