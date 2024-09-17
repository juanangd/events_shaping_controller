

class SmooothTrajectory:

    def __init__(self, frequency, smoothing_time):

        self.period = 1 / frequency

        # define function time intervals
        self.time_constant_half = self.period * (1 - smoothing_time) * 0.5
        self.smoothing_time = (smoothing_time / 4) * self.period
        self.t1 = self.smoothing_time

        self.init_curve = lambda t : 0 + (3 / self.smoothing_time ** 2) * t ** 2 - (2 / self.smoothing_time ** 3) * t ** 3


class SmoothConstantTrajectory(SmooothTrajectory):
    
    def __init__(self, smoothing_time = 0.1):
        """ smoothing time to reach the constant velocity"""
        super().__init__( 1/(smoothing_time * 4), 1.)

    def signal_at(self, time):

        if time < self.t1:
            return self.init_curve(time)
        else:
            return 1.

class SmoothedConstantCyclicTrajectory(SmooothTrajectory):
    """
    This will create a signal goes smoothly from 0 to 1, then from 1 to -1 and finally back to 0. This is
    a period of the signal.

    frequency: 1/T where T is the period of the signal
    smoothing_time: the sum of all the smoothing time (which in total are 4)
    """
    def __init__(self, frequency, smoothing_time):

        super().__init__(frequency, smoothing_time)
        
        self.t2 = self.t1 + self.time_constant_half
        self.t3 = self.t2 + 2 * self.smoothing_time
        self.t4 = self.t3 + self.time_constant_half

        # For more info refer to the docuent RO2.T.004-trajectories.part2.pdf from RBO robotics course.
        self.down_to_up_curve = lambda t : -1 + (3 / (self.smoothing_time * 2) ** 2) * 2 * t ** 2 - (2 / (self.smoothing_time * 2) ** 3) * 2 * t ** 3
        self.up_to_down_curve = lambda t: 1 + (3 / (self.smoothing_time * 2) ** 2) * -2 * t ** 2 - (2 / (self.smoothing_time * 2) ** 3) * -2 * t ** 3

    def signal_at(self, time):

        time_in_current_period = time % self.period
        current_period = time // self.period

        if time_in_current_period < self.t1:
            if current_period==0 :
                return self.init_curve(time_in_current_period)
            else:
                return self.down_to_up_curve(time_in_current_period + self.smoothing_time)

        elif self.t1 <= time_in_current_period < self.t2:
            return 1.
        elif self.t2 <= time_in_current_period < self.t3:
            return self.up_to_down_curve(time_in_current_period -self.t2)
        elif self.t3 <= time_in_current_period < self.t4:
            return -1.
        elif time_in_current_period >= self.t4:
            return self.down_to_up_curve(time_in_current_period - self.t4)
