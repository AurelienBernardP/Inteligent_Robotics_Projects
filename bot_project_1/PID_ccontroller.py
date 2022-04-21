
class PID_controller():

    def __init__(self, dt,kp,kd,ki):
        self.integral_effect = 0.0

        self.dt = dt
        self.previous_error = 0

        self.kp = kp
        self.kd = kd
        self.ki = ki

    def control(self,reference,current_val):

        current_error = current_val - reference

        self.integral_effect += (current_error * self.dt)
        derivative = (self.previous_error - current_error) / self.dt 

        self.previous_error = current_error

        return current_error * self.kp + derivative * self.kd + self.integral_effect * self.ki