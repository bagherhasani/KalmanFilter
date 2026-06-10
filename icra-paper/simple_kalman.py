import numpy as np 
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt


class PedestrainKalmanFilter:
    def __init__(self,dt):

        #dt is the time interval between measurements(e.g: can be camera frame rate)

        # State: x = [px, py, vx, vy]
        self.dt=dt
        self.kf=KalmanFilter(4,2)

       
        # Initial state: [px, py, vx, vy]
        self.kf.x = np.array([0.0, 0.0, 0.0, 0.0])


        #State Transition Matrix
        """
        px_new = px + vx * dt
        py_new = py + vy * dt
        vx_new = vx
        vy_new = vy
        """
        self.kf.F = np.array([
            [1.0, 0.0, self.dt, 0.0],  # px_new = 1*px + 0*py + dt*vx + 0*vy
            [0.0, 1.0, 0.0, self.dt],  # py_new = 0*px + 1*py + 0*vx + dt*vy
            [0.0, 0.0, 1.0, 0.0],      # vx_new = 0*px + 0*py + 1*vx + 0*vy
            [0.0, 0.0, 0.0, 1.0],      # vy_new = 0*px + 0*py + 0*vx + 1*vy
        ])

        # Measurement matrix H
        # zed gives  position: [px, py]
        # It does NOT directly measure velocity meaning that we have to calculate it ourself 
        self.kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # Initial uncertainty matrix P
        # Large values mean: "At the beginning, not truest to initial guess much."
        self.kf.P = np.eye(4) * 100.0
        """
        eye*4 makes below

        [100, 0,   0,   0]
        [0,   100, 0,   0]
        [0,   0,   100, 0]
        [0,   0,   0,   100]
        """

        # Measurement noise matrix R
        # This tells the filter how much noise we expect from the ZED position measurement.
        # Smaller value = trust ZED more
        # Larger value = trust ZED less
        self.kf.R = np.array([
            [0.05, 0.0],
            [0.0, 0.05],
        ])


        
         # Process noise matrix Q
        # This tells the filter how much the motion model can be wrong.
        # Larger velocity noise means the person is allowed to change speed/direction more.
        self.kf.Q = np.array([
            [0.01, 0.0,  0.0,  0.0],
            [0.0,  0.01, 0.0,  0.0],
            [0.0,  0.0,  0.1,  0.0],
            [0.0,  0.0,  0.0,  0.1],
        ])


    def update(self, measured_position):
        """
        measured_position = [px, py]
        This comes from ZED position measurement.
        """

        # Convert measurement to numpy array
        z = np.array(measured_position)

        # Step 1: predict where the person should be now
        self.kf.predict()

        # Step 2: correct prediction using ZED measurement
        self.kf.update(z)

        # Return current estimated state: [px, py, vx, vy]
        return self.kf.x
    



    def predict_future(self, seconds_ahead, step_dt):
        """
        Predict future pedestrian positions using the current Kalman state.

        Returns:
            list of [future_px, future_py]
        """

        px, py, vx, vy = self.kf.x

        future_positions = []

        t = step_dt
        while t <= seconds_ahead:
            future_px = px + vx * t
            future_py = py + vy * t

            future_positions.append([float(future_px), float(future_py)])

            t += step_dt

        return future_positions
    

if __name__ == "__main__":
    tracker = PedestrainKalmanFilter(dt=0.1)

    # Fake ZED position measurements: [px, py]
    measurements = [
    [0.00, 0.00],
    [0.10, 0.02],
    [0.21, -0.01],
    [0.32, 0.03],
    [0.43, 0.01],
    [0.55, -0.02],
    [0.66, 0.00],
    [0.78, 0.04],
    [0.89, 0.02],
    [1.01, -0.01],
    [1.12, 0.01],
    [1.24, 0.03],
    [1.35, 0.00],
    [1.47, -0.02],
    [1.58, 0.01],
    [1.70, 0.04],
    [1.82, 0.02],
    [1.93, -0.01],
    [2.05, 0.01],
    [2.16, 0.03],
]

    # Store data for plotting
    measured_x = []
    measured_y = []

    filtered_x = []
    filtered_y = []

    final_future_positions = []

    for measurement in measurements:
        state = tracker.update(measurement)

        px, py, vx, vy = state

        measured_x.append(measurement[0])
        measured_y.append(measurement[1])

        filtered_x.append(float(px))
        filtered_y.append(float(py))

        final_future_positions = tracker.predict_future(
            seconds_ahead=1.0,
            step_dt=0.2
        )

        print("----------------")
        print("ZED measurement:", measurement)
        print("Kalman state [px, py, vx, vy]:", state)
        print("future positions:", final_future_positions)

    # Split future positions into x and y lists
    future_x = [p[0] for p in final_future_positions]
    future_y = [p[1] for p in final_future_positions]

    # Current final Kalman position
    current_x = filtered_x[-1]
    current_y = filtered_y[-1]

    # Plot everything
    plt.figure(figsize=(10, 6))

    # ZED measurements
    plt.scatter(measured_x, measured_y, label="ZED measurements")

    # Kalman filtered trajectory
    plt.plot(filtered_x, filtered_y, marker="o", label="Kalman filtered path")

    # Future predicted trajectory
    plt.plot(future_x, future_y, marker="x", linestyle="--", label="Future prediction")

    # Current final position
    plt.scatter([current_x], [current_y], s=120, label="Current estimated position")

    plt.title("Pedestrian Kalman Filter: Measurement, Filtered Path, and Future Prediction")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()