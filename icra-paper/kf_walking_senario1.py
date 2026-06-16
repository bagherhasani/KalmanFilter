"""
In this example the problem is: 
    Given the current position of a walking person, estimate their walking speed,
    and predict where they will be in near future.

Relattinship to paper: 
    A person is walking toward robot, 
    Zed camera gives the person's current distance from the robot at each time step,
    I want to estimate how fast the person is approching and estimate their future distance.

Input example: 
    z = 5.0 meters 
    z = 4.7 meters
    z = 4.3 meters 
    z = 4.1 meters


"""



from cProfile import label

from filterpy.kalman import KalmanFilter
import numpy as np 
import matplotlib.pyplot as plt


class KF:
    def __init__(self,dt,initial_position, initial_vel):

        self.initial_position=initial_position
        self.initial_vel=initial_vel
        self.dt=dt


        # initialize kf sizes 
        self.kf=KalmanFilter(2,1)

        # define x : state vector
        self.kf.x=np.array([self.initial_position, self.initial_vel])



        # defin F: motion model. we put matrix form not equations
        # newpos = oldpos+(vel*dt)
        # newvel = oldvel
        self.kf.F=np.array([
                [1 , self.dt],
                [0 ,       1]
                           ])
        

        # Define H: measurement function here tell the kf that what parts can sensor measure
        # my full state is x =[pos,vel]
        # zed gives me only the [pos]
        # H connects state to measurement
        self.kf.H=np.array([[1,0]])



        """
        Define P: how kf uncertain I am about my initial state
        P = [
            [position-position uncertainty,   position-velocity relationship],
            [velocity-position relationship, velocity-velocity uncertainty]]
        """
        self.kf.P=np.array(
            [
                [1,0],[0,100]
            ]
        )

        """
        Define R: Reasurement noise - how noisy the zed-camera might be ? 
        """
        self.kf.R=np.array([[0.1]])


        """Define Q: Process noise / motion model uncertainity
            Meaning that how imperfect my motion model is 
            Q allows motion model to change a little bit or alot because we do not trust constant velocity"""
        self.kf.Q=np.array(
            [[0.01,0],  
             [0,0.1]]
             #position process noise = very small velocity / process noise = bigger, because people can speed up/slow down
        )
    

    def predict(self):
        self.kf.predict()
        return self.kf.x
        
    



    def update(self,measured_position):
        self.kf.update(np.array([measured_position]))

        return self.kf.x


    def process_measurement(self,measured_position):
        self.predict()
        self.update(measured_position)

        position=float(self.kf.x.flatten()[0])
        velocity= float(self.kf.x.flatten()[1])


        return position,velocity
    


    def predict_future(self,seconds_ahead):
        pos=float(self.kf.x.flatten()[0])
        vel=float(self.kf.x.flatten()[1])

        future_pos=pos+(vel*seconds_ahead)

        return future_pos


if __name__=="__main__":

    kf=KF(0.1,5.0,0.0)

    measurements = [
    5.00, 4.97, 4.93, 4.90, 4.86,
    4.83, 4.79, 4.76, 4.72, 4.69,
    4.65, 4.62, 4.58, 4.55, 4.51,
    4.48, 4.44, 4.41, 4.37, 4.34,
    4.30, 4.27, 4.23, 4.20, 4.16,
    4.13, 4.09, 4.06, 4.02, 3.99,
    3.95, 3.92, 3.88, 3.85, 3.81,
    3.78, 3.74, 3.71, 3.67, 3.64,
    3.60, 3.57, 3.53, 3.50, 3.46,
    3.43, 3.39, 3.36, 3.32, 3.29,
    3.25, 3.22, 3.18, 3.15, 3.11,
    3.08, 3.04, 3.01, 2.97, 2.94
]

    acutal=[]
    kf_esitmated=[]
    future_prediction=[]
    seconds_ahead=0.5

    for item in measurements:
        pos,vel=kf.process_measurement(item)
        print(f"measured is:{item} esitmated pos:{pos} estimated vel:{vel}")
        print("-------------")

        #plot
        acutal.append(item)
        kf_esitmated.append(pos)

        #future pos
        future_pos=kf.predict_future(seconds_ahead)
        future_prediction.append(future_pos)


    time_steps = list(range(len(measurements)))

    steps_ahead = int(seconds_ahead / kf.dt)
    future_time_steps = [t + steps_ahead for t in time_steps]

    plt.plot(time_steps, acutal, label="actual / zed raw")
    plt.plot(time_steps, kf_esitmated, label="kf-estimated")
    plt.plot(
        future_time_steps,
        future_prediction,
        label=f"future prediction {seconds_ahead}s ahead"
    )

    plt.xlabel("Time step")
    plt.ylabel("Distance from robot (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

   
