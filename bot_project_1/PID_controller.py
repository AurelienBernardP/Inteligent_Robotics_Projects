import numpy as np
import matplotlib.pyplot as plt
class PID_stats():

    def __init__(self):

        self.ref_val = []
        self.current_val = []

    def update(self,reference, cur_val):
        self.ref_val.append(reference)
        self.current_val.append(cur_val)
        return

    def plot(self,ki,kd,kp,dt):

        time_scale = np.arange(0,len(self.ref_val)*dt,step= dt)

        plt.plot(time_scale,self.ref_val,color='green',linestyle='dashed')
        plt.plot(time_scale,self.current_val,color='red')
        plt.title(('PID signals with kp = ' + str(kp) + ' ki = ' + str(ki) + ' kd = ' + str(kd)))
        plt.legend()
        plt.show()


        

            

        
class PID_controller():

    def __init__(self, dt,kp,kd,ki,stats=False):
        self.integral_effect = 0.0

        self.dt = dt
        self.previous_error = 0
        self.previous_reference = 0
        self.kp = kp
        self.kd = kd
        self.ki = ki

        self.previous_actuator_speed = 0.0
        self.PID_statistics = None
        if stats == True:
            self.PID_statistics = PID_stats()

    def control(self,reference,current_val):

        current_error =  reference - current_val
        
        derivative = (current_error - self.previous_error) / self.dt 
        if self.previous_reference != reference:
            derivative = 0
            self.previous_reference = reference
            
        current_speed = derivative

        if(current_speed != self.previous_actuator_speed):
            #this statement prevents integration when the actuator saturates to prevent overshoots
            self.integral_effect += (current_error * self.dt)
            self.previous_actuator_speed = current_speed

        self.previous_error = current_error

        if(self.PID_statistics != None):
            self.PID_statistics.update(reference,current_val)
        return current_error * self.kp + derivative * self.kd + self.integral_effect * self.ki


    def plot(self):
        if self.PID_statistics == None:
            print("no statistics were recorded for this PID")
            return

        self.PID_statistics.plot(self.ki,self.kd,self.kp,self.dt)
