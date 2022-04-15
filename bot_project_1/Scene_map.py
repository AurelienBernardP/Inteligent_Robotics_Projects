import numpy as np
import matplotlib.pyplot as plt


class Scene_map :

    UNEXPLORED = 0
    OBSTACLE = 1
    FREE = 2
    BOT = 3
    RAY = 4

    def __init__(self, width, height):
        self.map_size = (width,height)

        self.occupancy_matrix = np.zeros((height, width))
        
        self.bot_pos = np.zeros(2) #(x,y)
        
        self.ray_endings = [] # matrix of Nb_rays columns and each column = (x,y,z)

        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.line1 = self.ax.imshow(self.occupancy_matrix,vmin=0, vmax=5,cmap = 'binary')
        
        
    def set_cooridnate_state(self, x_y_position,state):
        if(state < 0 or state > 3):
            print("error")
            return
        
        if(x_y_position[0] < 0 or x_y_position[1] < 0 or x_y_position[0] >= self.map_size[0] or x_y_position[1] >= self.map_size[1]):
            print("error")
            return

        self.occupancy_matrix[x_y_position] = state
    
    def update_bot_pos(self, new_pos):
        self.bot_pos[0] = new_pos[0]
        self.bot_pos[1] = new_pos[1]

    def update_contact_map_from_sensor(self,points,occupancy):

        self.ray_endings = np.hstack((points[:3,:],points[3:,:]))

        for i in range(np.shape(self.ray_endings)[1]):
            self.ray_endings[0,i] += self.bot_pos[0]
            self.ray_endings[1,i] += self.bot_pos[1]

    def show_map_state(self):

        drawn_matrix = self.occupancy_matrix = np.zeros((np.shape(self.occupancy_matrix)[0], np.shape(self.occupancy_matrix)[1]))
        drawn_matrix[map_position_to_mat_index(self.bot_pos[0],self.bot_pos[1])] = 3

        print(np.shape(self.ray_endings))
        for i in range(np.shape(self.ray_endings)[1]):
            drawn_matrix[map_position_to_mat_index(self.ray_endings[0,i],self.ray_endings[1,i])] = 4

        self.line1.set_data(drawn_matrix)

        plt.xlim(0,np.shape(self.occupancy_matrix)[0])
        plt.ylim(0,np.shape(self.occupancy_matrix)[1])

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


def map_position_to_mat_index(x,y):
    return int(y*10 + 75),int(x*10 + 75) 