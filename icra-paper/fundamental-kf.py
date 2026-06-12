from filterpy.kalman import KalmanFilter
import numpy as np


class KF:
    def __init__(self):

        #initialize kf
        self.kf=KalmanFilter(1,1)

        #initial guess 
        # room tempreture is 20 at first
        self.kf.x=np.array([[20.0]])

        # F = state transition model 
        # Temp-new = temp_old
        # we assume the temp does not change quickly
        self.kf.F=np.array([[1.0]])

        #H = measurement model
        # sensor directly measure temp
        self.kf.H=np.array([[1.0]])

        #P = uncertainity in our initial guess
        # Large P means I am not very sure about the inital guess
        self.kf.P=np.array([[1.0]])

        # R = measurement noise 
        # Larger R means the sensor is noisy ; do not trust it too much
        self.kf.R=np.array([[1.0]])

        # Q = process noise 
        # smaller Q means the temp changes slowly.
        self.kf.Q=np.array([[0.01]])

    def update(self,measurement):
        self.kf.predict()
        self.kf.update(np.array([measurement]))

        return float(self.kf.x.flatten()[0])


if __name__=="__main__":
    kf=KF()

    measurements=[20.4, 19.7, 20.2, 21.0, 19.8, 20.1, 20.0]

    for item in measurements:
        estimate=kf.update(item)


        print("-----------------")
        print("noisy measurement",item)
        print("kalman filter estimate",estimate)
       
        
      

