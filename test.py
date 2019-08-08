from DeepKalmanFilter import DeepKalmanFilter
from pykalman import KalmanFilter
import numpy as np

def simulateLinearData(N, T, DIM):
    """ Synthetic data generated according to a first order linear Markov process """
    z    = np.random.randn(N, DIM)
    zlist= [np.copy(z)[:,None,:]]
    W    = 0.1*np.random.randn(DIM,DIM)
    for t in range(T-1):
        z_next = np.dot(z,W) 
        zlist.append(np.copy(z_next)[:,None,:])
        z      = z_next
    Z   = np.concatenate(zlist, axis=1)
    X   = Z + 4*np.random.randn(*Z.shape) 
    return X, Z

def get_next_batch(Data, batch_count, batch_num = 32):
    if (batch_count + batch_num > Data.shape[0]):
        batch_count = (batch_count + batch_num) % Data.shape[0]
    batch_data = Data[batch_count: batch_count+batch_num]
    batch_count = (batch_count + batch_num) % Data.shape[0]
    return batch_data, batch_count

if __name__ == "__main__":
    obs_count = state_count = 0
    data_obs, data_states = simulateLinearData(10000, 10, 3)

    DKF = DeepKalmanFilter(3, 3, 10)

    for i in range(2000):
        batch_obs, obs_count= get_next_batch(data_obs, obs_count)
        batch_state, state_count = get_next_batch(data_states, state_count)

        DKF.train_model(batch_obs)

        if i%20 == 0:
            RMSE, ELBO = DKF.test_model(batch_state, batch_obs)
            print("At iteration %d: ELBO\t%f RMSE\t%f" %(i, ELBO, RMSE))

    # transition_offset = 0.05 * np.ones(shape=(3))
    # transition_covariance = 10*np.eye(3)
    # observation_mat = 0.5 * np.eye(3)
    # observation_covariance = 20 * np.eye(3)
    # KF = KalmanFilter(
    #     transition_offsets = transition_offset,
    #     transition_covariance = transition_covariance,
    #     observation_matrices = observation_mat,
    #     observation_covariance = observation_covariance
    # )
    # RMSE = []
    # for i in range(1000):
    #     smoothed_state = KF.smooth(data_obs[i])[0]
    #     rmse = np.mean(np.sqrt(np.square(smoothed_state - data_states[i])))
    #     RMSE.append(rmse)