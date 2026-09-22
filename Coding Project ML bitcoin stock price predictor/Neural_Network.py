import numpy as np
import Data_Pipeline
import os

class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        
        # Initailly creates a matrix with random weights 
        
        #Sets the matrix with the biases to 0 initially
        if weights is None:
            self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
             # Initailly creates a matrix with random weights if weights have not been defined 
        else:
            self.weights = weights
        if biases is None :
            self.biases = np.zeros((1, n_neurons))
             #Sets the matrix with the biases to 0 initially
        else :
            self.biases = biases
        self.V_weights = np.zeros_like(self.weights)
        self.V_biases = np.zeros_like(self.biases)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
    
    def backwards(self, dldz):
        self.dweights = np.dot(self.inputs.T, dldz)
        self.dbiases = np.sum(dldz, axis=0, keepdims=True)
        self.dinputs = np.dot(dldz, self.weights.T)
        #print("Max Gradient:", np.max(np.abs(self.dweights)))  # Check gradients size
    
#   Momentum gradient decent

    def Velocity_Update(self, Y, n):
        self.V_weights = (Y*self.V_weights)-(n*self.dweights)
        self.V_biases = (Y*self.V_biases)-(n*self.dbiases)
    
    def Parameter_Update(self):
        self.weights += self.V_weights
        self.biases += self.V_biases
    
    def Store_Weights(self):
        weights_array = np.array(self.weights) 
        return weights_array          
        
    def Store_Biases(self):
        biases_array = np.array(self.biases) 
        return biases_array

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backwards(self, derivatives):
        self.dinputs = derivatives.copy()
        self.dinputs[self.inputs <= 0] =0
        

class MSE_Loss:
    def forward(self, Predicted, Actual):
        Y_Predict = np.array(Predicted)
        Y_Actual = np.array(Actual)
        loss = np.mean((Y_Predict-Y_Actual)**2)
        self.output = loss
        #print(loss)
    
    def backwards(self, Predicted, Actual):
        Predicted = np.array(Predicted).reshape(-1,1)
        Actual = np.array(Actual).reshape(-1,1)

        n = len(Predicted)
        self.dinputs = 2/n*(Predicted-Actual)


def Testing_loop(X, Y):
    Loss_list = []
    for t in range(len(X)):
        Target = Y[t]
        Input_Layer.forward(X[t])
        RELU.forward(Input_Layer.outputs)
        Hidden_Layer_1.forward(RELU.output)
        RELU2.forward(Hidden_Layer_1.outputs)
        Hidden_Layer_2.forward(RELU2.output)
        RELU3.forward(Hidden_Layer_2.outputs)
        Output_Layer.forward(RELU3.output)
        Loss = MSE_Loss()
        Loss.forward(Output_Layer.outputs, Target)
        Loss_list.append(Loss.output)
    RMSE = np.sqrt(np.mean(Loss_list)) * Nea_Stock_Adding.std_dev_train_Y
    return RMSE
   

learning_rates = [0.05, 0.01]
momentums = [0.3, 0.5, 0.7]
Training_Loops = [3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200]

def best_parameters(learning_rates, momentums, Training_Loops, X_train, Y_train, X_val, Y_val, X_test, Y_test):
    best_val_rmse = float('inf')
    best_test_rmse = float('inf')
    for lr in learning_rates:
        for m in momentums:
            for TL in Training_Loops:
                print(f"Result of parameters : learning rate = {lr}, momentum = {m} and training loop = {TL}")
                
                for i in range(TL):
                    #forward pass
                    Input_Layer.forward(X_train) 
                    RELU.forward(Input_Layer.outputs) 
                    Hidden_Layer_1.forward(RELU.output) 
                    RELU2.forward(Hidden_Layer_1.outputs) 
                    Hidden_Layer_2.forward(RELU2.output) 
                    RELU3.forward(Hidden_Layer_2.outputs) 
                    Output_Layer.forward(RELU3.output) 
                     
                    #loss 
                    Loss = MSE_Loss()
                    Loss.forward(Output_Layer.outputs, Y_train)
                    Loss.backwards(Output_Layer.outputs, Y_train) 
                     
                    Output_Layer.backwards(Loss.dinputs) 
                    RELU3.backwards(Output_Layer.dinputs) 
                    Hidden_Layer_2.backwards(RELU3.dinputs) 
                    RELU2.backwards(Hidden_Layer_2.dinputs) 
                    Hidden_Layer_1.backwards(RELU2.dinputs) 
                    RELU.backwards(Hidden_Layer_1.dinputs) 
                    Input_Layer.backwards(RELU.dinputs)      
                    Input_Layer.Velocity_Update(m, lr) 
                    Input_Layer.Parameter_Update() 
                    Hidden_Layer_1.Velocity_Update(m, lr) 
                    Hidden_Layer_1.Parameter_Update() 
                    Hidden_Layer_2.Velocity_Update(m, lr) 
                    Hidden_Layer_2.Parameter_Update() 
                    Output_Layer.Velocity_Update(m, lr)
                
                Val_RMSE = Testing_loop(X_val, Y_val)
                Test_RMSE = Testing_loop(X_test, Y_test)
                print(f"Training loop: {TL}, Validation RMSE: {Val_RMSE}, Test RMSE: {Test_RMSE}")
                layers = [Input_Layer, Hidden_Layer_1, Hidden_Layer_2, Output_Layer]
                model = "1 Year Model"
                
                
                if Val_RMSE < best_val_rmse and Test_RMSE < best_test_rmse:
                    best_val_rmse = Val_RMSE
                    best_test_rmse = Test_RMSE
                    os.makedirs(model, exist_ok=True)
                    for i, layer in enumerate(layers, start=1):
                        np.save(f"{model}/layer{i}_weights.npy", layer.Store_Weights())
                        np.save(f"{model}/layer{i}_biases.npy", layer.Store_Biases())
                    
                print(f"Training loop: {TL}, best Validation RMSE: {best_val_rmse}, best Test RMSE: {best_test_rmse}")

Learning_rate = 0.05
Momentum = 0.7 
Training_Loop = 4300

Input_Layer = Layer(9, 10) 
Hidden_Layer_1 = Layer(10, 12) 
Hidden_Layer_2 = Layer(12, 12) 
Output_Layer = Layer(12, 1) 
RELU = Activation_ReLU() 
RELU2 = Activation_ReLU() 
RELU3 = Activation_ReLU() 
