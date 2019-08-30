import tensorflow as tf
import numpy as np

class DeepKalmanFilter:

    def __init__(self, n_dim_state, n_dim_obs, n_time_steps, batch_num = 32):
        self.n_dim_state = n_dim_state 
        self.n_dim_obs = n_dim_obs
        self.n_time_steps = n_time_steps
        self.batch_num = batch_num

        self.lr = 1e-3
        
        self.build_model()

    def train_model(self, batch_observations):
        feed_dict = {self.input_obs: batch_observations}
        self.sess.run(self.train_step, feed_dict = feed_dict)

    def test_model(self, batch_states, batch_observations):
        '''
        Test Inference 
        '''
        RMSE, ELBO = self.sess.run([self.RMSE, self.ELBO], feed_dict = {self.input_obs: batch_observations, self.input_states: batch_states})
        return RMSE, ELBO

    def build_model(self):
        # input_state = tf.placeholder(tf.float32, shape=(None, self.n_dim_state))
        # '''
        # Transition Function (same transition function for each time step): Gated Transition Function
        # '''
        # state_means = []
        # state_logvars = []
        # for i in range(self.n_time_steps):
        #     next_mean, next_logvar = self.transition(input_state)
        #     state_means.append(next_mean)
        #     state_logvars.append(next_logvar)

        # '''
        # Emission Function (same emission function for each time step): 3 layer MLP
        # ''' 
        # obs_means = []
        # obs_logvars = []
        # for i in range(self.n_time_steps):
        #     tmp_mean, tmp_logvar = self.emission(input_state)
        #     obs_means.append(tmp_mean)
        #     obs_logvars.append(tmp_logvar)

        '''
        Inference Network
        '''
        self.input_obs = tf.placeholder(tf.float32, shape=(self.batch_num, self.n_time_steps, self.n_dim_obs))
        # input hidden states for testing
        self.input_states = tf.placeholder(tf.float32, shape=(self.batch_num, self.n_time_steps, self.n_dim_state))
        
        forward = tf.keras.layers.SimpleRNN(units = 64, return_sequences = True)
        forward_2 = tf.keras.layers.SimpleRNN(units = 64, return_sequences = True)
        backward = tf.keras.layers.SimpleRNN(units = 64, return_sequences = True, go_backwards = True)
        backward_2 = tf.keras.layers.SimpleRNN(units = 64, return_sequences = True, go_backwards = True)

        h_forward = tf.unstack(forward_2(forward(self.input_obs)), axis=1)
        h_backward = tf.unstack(backward_2(backward(self.input_obs)), axis=1)
        
        self.infer_state_means = []
        self.infer_state_logvars = []
        self.infer_states = []
        
        current_state = tf.zeros(shape=(self.batch_num, self.n_dim_state))
        for i in range(self.n_time_steps):
            tmp_h_f = h_backward[i]
            tmp_h_b = h_backward[i]
            
            tmp_state_mean, tmp_state_logvar = self.combiner(current_state, tmp_h_f, tmp_h_b)
            current_state = self.reparameterize(tmp_state_mean, tmp_state_logvar)
            
            self.infer_states.append(current_state)

            self.infer_state_means.append(tmp_state_mean) 
            self.infer_state_logvars.append(tmp_state_logvar)

        '''
        Define train loss
        '''
        infer_obses = tf.unstack(self.input_obs, axis=1)
        P_X_Z = 0
        for i in range(self.n_time_steps):
            tmp_mean, tmp_logvar = self.emission(self.infer_states[i])
            P_X_Z += self.log_normal_pdf(infer_obses[i], tmp_mean, tmp_logvar)
        
        KLD = self.log_normal_pdf(self.infer_states[0], 0., 0.) - self.log_normal_pdf(self.infer_states[0], self.infer_state_means[0], self.infer_state_logvars[0])
        for i in range(self.n_time_steps-1):
            tmp_mean, tmp_logvar = self.transition(self.infer_states[i])
            KLD += self.log_normal_pdf(self.infer_states[i+1], tmp_mean, tmp_logvar)
            KLD -= self.log_normal_pdf(self.infer_states[i+1], self.infer_state_means[i+1], self.infer_state_logvars[i+1])

        self.ELBO = tf.reduce_mean(P_X_Z + KLD)
        self.loss = -self.ELBO

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        '''
        Define testing RMSE
        '''
        infer_states = tf.stack(self.infer_states, axis = 1)
        # infer_states = tf.reshape(infer_states, shape = [self.batch_num, self.n_time_steps, self.n_dim_state])
        self.RMSE = tf.reduce_mean(tf.math.sqrt(tf.square(self.input_states - infer_states)))
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    '''
    Model Components
    '''
    def emission(self, input_state):
        '''
        input_state: (None, n_dim_state)
        Output: 
            output_obs: (None, n_dim_obs)
            logvar_obs: (None, n_dim_obs)
        '''
        with tf.variable_scope("Emission_Layer_1", reuse = tf.AUTO_REUSE):
            current = self.fc_layer(input_state, self.n_dim_state, 64)
        with tf.variable_scope("Emission_Layer_2", reuse = tf.AUTO_REUSE):
            current = self.fc_layer(current, 64, 128)
        with tf.variable_scope("Emission_Layer_3", reuse = tf.AUTO_REUSE):
            Wfc = self.weight_variable_xavier([ 128, self.n_dim_obs*2 ], name = 'W')
            bfc = self.bias_variable([ self.n_dim_obs*2 ])
            current = tf.matmul(current, Wfc) + bfc
            mean_obs, logvar_obs = tf.split(current, num_or_size_splits=2, axis=1)
        return mean_obs, logvar_obs

    def transition(self, state_pre):
        '''
        Input: (None, n_dim_state)
        Output:
            state_t+1_mean: (None, n_dim_state)
            state_t+1_logvar: (None, n_dim_state)
        We implement a gated transtion function here
        '''
        # Gating Unit
        with tf.variable_scope("Gating_Unit_Layer_1", reuse = tf.AUTO_REUSE):
            current = self.fc_layer(state_pre, self.n_dim_state, 128)
        with tf.variable_scope("Gating_Unit_Layer_2", reuse = tf.AUTO_REUSE):
            gate = self.fc_layer(current, 128, self.n_dim_state, activation = "sigmoid")

        # Proposed Mean
        with tf.variable_scope("Proposed_Mean_Layer_1", reuse = tf.AUTO_REUSE):
            current = self.fc_layer(state_pre, self.n_dim_state, 128)
        with tf.variable_scope("Proposed_Mean_Layer_2", reuse = tf.AUTO_REUSE):
            mean_h = self.fc_layer(current, 128, self.n_dim_state, activation = "I")

        # Transiton 
        with tf.variable_scope("Real_Transition", reuse = tf.AUTO_REUSE):
            # mean
            W_mean = self.weight_variable_xavier([ self.n_dim_state, self.n_dim_state ], name = 'W_mean')
            b_mean = self.bias_variable([ self.n_dim_state ], name = "b_mean")
            mean_next = (1-gate)*(tf.matmul(state_pre, W_mean) + b_mean) + gate * mean_h
            # variance
            W_var = self.weight_variable_xavier([ self.n_dim_state, self.n_dim_state ], name = 'W_var')
            b_var = self.bias_variable([ self.n_dim_state ], name = "b_var")
            logvar_next = self.softplus(tf.matmul(tf.nn.relu(mean_h), W_var) + b_var)
            
        return mean_next, logvar_next

    def combiner(self, z_pre, h_t_left, h_t_right):
        with tf.variable_scope("Combiner_Layer", reuse = tf.AUTO_REUSE):
            W_c = self.weight_variable_xavier([ self.n_dim_state, 64 ], name = 'W_combiner')
            b_c = self.bias_variable([ 64 ], name = "b_Combiner")
            h_combined = 1/3 * (tf.tanh(tf.matmul(z_pre, W_c) + b_c) + h_t_left + h_t_right)

            W_mean = self.weight_variable_xavier([ 64, self.n_dim_state ], name = 'W_mean')
            b_mean = self.bias_variable([ self.n_dim_state ], name = "b_mean")
            infer_mean = tf.matmul(h_combined, W_mean) + b_mean

            W_var = self.weight_variable_xavier([ 64, self.n_dim_state ], name = 'W_var')
            b_var = self.bias_variable([ self.n_dim_state ], name = "b_var")
            infer_logvar = self.softplus(tf.matmul(h_combined, W_var) + b_var)
        return infer_mean, infer_logvar

    '''
    Helper Funtions
    '''
    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name = 'bias'):
        initial = tf.constant(0.0, shape = shape)
        return tf.get_variable(name = name, initializer = initial)

    def fc_layer(self, current, in_features, out_features, activation = "relu"):
        Wfc = self.weight_variable_xavier([ in_features, out_features ], name = 'W')
        bfc = self.bias_variable([ out_features ])
        current = tf.matmul(current, Wfc) + bfc
        if activation == "relu":
            current = tf.nn.relu(current)
        if activation == "sigmoid":
            current = tf.nn.sigmoid(current)
        return current
    
    def softplus(self, X):
        return tf.math.log((1+tf.math.exp(X)))

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
            
    # def reparameterize(self, input_mean, input_logvar):
    #     mean = tf.placeholder(tf.float32)
    #     logvar = tf.placeholder(tf.float32)
    #     eps = tf.random.normal(shape=input_mean.shape)
    #     return self.sess.run(eps * tf.exp(logvar * .5) + mean, \
    #         feed_dict = {mean: input_mean, logvar: input_logvar})
    
if __name__ == "__main__":
    model = DeepKalmanFilter(10, 20, 5)