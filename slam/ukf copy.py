import numpy as np
import math

class UKF:
    
    def predict(motion_model: callable, x, P, Q):
        """ Executes prediction step for the UKF. This step happens before the arrival of measurement data
        :type motion_model:callable: the motion model method should receive a state and return the next state, based on applying the laws of motion
        :param motion_model:callable: x

        :type x: vehicle vector state with dimension N. Can be [position velocity] for example
        :param x: last state

        :type P: covariance matrix (NxN)
        :param P: Last valid covariance matrix

        :type Q: model error
        :param Q: last model error

        :raises:

        :rtype: next predicted state (1xN), next predicted covariance matrix (NxN)
        """

        pd = np.linalg.cholesky(P)
        
        len_x = UKF.__len(x)
        
        N = len_x
        k = 3 - N
        SP = np.zeros((N, 2*N + 1))

        q = math.sqrt(N + k)

        SP[:,0] = x
        for i in range(1, N+1):
            SP[:,i] = x + q * pd.T[i-1]
            SP[:,i+N] = x - q * pd.T[i-1]

        a0 = k / (k + N)
        a1 = 0.5 * (1/(k+N))

        Xp = 0
        Pp = Q
        
        SP[:,0] = motion_model(SP[:,0])
        Xp = a0 * SP[:,0]
        for i in range (1, 2*N+1):
            SP[:, i] = motion_model(SP[:,i])
            Xp += a1 * SP[:,1]

        for i in range (0, 2*N+1):
            a = a0
            if i > 0: a = a1    
            diff = SP[:,i] - Xp
            diff = np.atleast_2d(diff)
            Pp += a * np.dot(diff.T, diff)

        return Xp, Pp
    
    def __len(val) -> int:
        if np.isscalar(val): return 1
        return len(val)
    
    def correct(measurement_model: callable, Xp, Pp, y, R):
        # predict the measurement
        pd = np.linalg.cholesky(Pp)
        
        len_x = UKF.__len(Xp)
        len_y = UKF.__len(y)

        N = len_x
        k = 3 - N
        SPx = np.zeros((len_x, 2*N + 1))       
        SPy = np.zeros((len_y, 2*N + 1))
        q = math.sqrt(N + k)

        SPx[:,0] = Xp
        for i in range(1, N+1):
            SPx[:,i] = Xp + q * pd.T[i-1]
            SPx[:,i+N] = Xp - q * pd.T[i-1]
        
        a0 = k / (k + N)
        a1 = 0.5 * (1/(k+N))
        
        # predict the mean
        SPy[:,0] = measurement_model(SPx[:,0])
        yp = a0 * SPy[:,0]
        for i in range (1, 2*N+1):
            SPy[:, i] = measurement_model(SPx[:,i])
            yp += a1 * SPy[0,1]
        
        # predict the covariance
        Py = R
        for i in range (0, 2*N+1):
            a = a0
            if i > 0: a = a1        
            diff = (SPy[:,i] - yp)
            Py += a * np.dot(diff, diff.T)
        
        # compute the cross-covariance
        Pxy = 0
        for i in range (0, 2*N+1):
            a = a0
            if i > 0: a = a1        
            
            diffX = (SPx[:,i] - Xp)
            #diffX = np.atleast_2d(diffX)
            
            diffY = (SPy[:,i] - yp)
            if UKF.__len(diffY) == 1:
                diffY = diffY[0]
            
            #diffY = np.atleast_2d(diffY)            
            
            Pxy += a * np.dot(diffX, diffY.T)
            #Pxy += a * diffX * diffY.T
        
        if UKF.__len(Py) == 1:
            K = Pxy * 1/Py
            X = Xp + K * (y - yp)
            P = Pp - K * Py @ K.T
        else:
            K = Pxy @ np.linalg.inv(Py)
            X = Xp + K @ (y - yp)
            P = Pp - K @ Py @ K.T
               
        return X, P
            
if __name__ == "__main__":
    x0 = np.array([0, 5])

    P0 = np.array([
    [0.00001, 0],
    [0, 1]
    ])
    Q = np.array([
            [0.01, 0],
            [0, 0.01]
        ])

    def motion_model(x: np.array) -> np.ndarray:
        A = np.array([
            [1, 0.5],
            [0, 1]
        ])
        return A @ x + -2 * np.array([0, 0.5])

    def meas_model(x) -> float:
        return math.atan(20 / (40 - x[0]))
        

    x1, P1 = UKF.predict(motion_model, x0, P0, Q)
                    

    X1c, P1c = UKF.correct(meas_model, x1, P1, math.pi/6, 0.01)
    
    print(f"estimate pos = {X1c[0]:.2f} m, velocity = {X1c[1]:.2f} m/s")
              
p = 1