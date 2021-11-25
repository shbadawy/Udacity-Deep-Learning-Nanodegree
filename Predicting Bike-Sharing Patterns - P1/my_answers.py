import numpy as np
import math

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
                                                        
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1/(1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        self.sigmoid_output_2_derivative = lambda x : x*(1-x)
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
            
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
#         self.lr *= (1. / (1. + 1e-6 * (iterations/10000))) 
    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
    
        hidden_inputs = X.dot(self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
         # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        
        final_inputs =  np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs =  final_inputs# signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        layer_1 = self.activation_function(X.dot(self.weights_input_to_hidden))
        layer_2 = layer_1.dot(self.weights_hidden_to_output)
        
        # TODO: Output error - Replace this value with your calculations.
        layer2_error = y-final_outputs #1-1
        layer2_delta = layer2_error
        
        layer1_error = layer2_delta.dot(self.weights_hidden_to_output.T) 
        layer1_delta = layer1_error * (self.sigmoid_output_2_derivative(layer_1))
        
        delta_weights_h_o += hidden_outputs.reshape((hidden_outputs.shape[0],1)).dot(layer2_delta.reshape((1,layer2_delta.shape[0])).T)
        delta_weights_i_h += X.reshape(X.shape[0],1)*layer1_delta
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += (delta_weights_h_o* self.lr) /n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += (delta_weights_i_h * self.lr) / n_records# update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
#         hidden_inputs = features.dot(self.weights_input_to_hidden) # signals into hidden layer
#         hidden_outputs = hidden_inputs # signals from hidden layer
#         # TODO: Output layer - Replace these values with your calculations.
#         final_inputs =  hidden_outputs.dot(self.weights_hidden_to_output)# signals into final output layer
#         final_outputs =  self.activation_function(final_inputs)
        
        final_outputs,_ = self.forward_pass_train(features)        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
# iterations = 1000
# learning_rate = 0.03
# hidden_nodes = 5
# output_nodes = 1
iterations = 2700
learning_rate = 0.5
hidden_nodes = 25
output_nodes = 1

# iterations = 2000
# learning_rate = 0.05
# hidden_nodes = 50
# output_nodes = 1



# iterations = 2000
# learning_rate = 0.07
# hidden_nodes = 5
# output_nodes = 1
