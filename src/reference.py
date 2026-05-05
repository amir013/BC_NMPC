import numpy as np

class Regulator():
    ''' creates a reference path depending on the knot points '''
    
    def __init__(self, h_controller):
        
        self.h = h_controller
        
    def reference(self, duration, knot_points, ref_type='zoh'): # zoh: zero-order-hold
    
        N = int(duration/self.h) + 1
        
        time_knots = knot_points['t']
        N_time = len(time_knots)
        time = np.linspace(0, duration, N)
        
        keys = list(knot_points.keys())
        keys.remove('t')
        
        for key in keys:
            set_points = knot_points[key]
            reference = np.ones((N_time, N))
            for index, val in enumerate(set_points):
                reference[index,:] *= val
        
            knot_points[f'{key}_ref'] = reference
            
        intervals = np.searchsorted(time_knots, time, side="right") - 1
        intervals = np.clip(intervals, 0, len(time_knots) - 1)
        knot_points['time_interval'] = intervals
        knot_points['time'] = time
        
        return knot_points