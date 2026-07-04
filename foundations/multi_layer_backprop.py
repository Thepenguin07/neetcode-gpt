import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        
        # 1. Convert input lists to NumPy arrays for vectorized mathematical operations
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        # ==========================================
        # FORWARD PASS
        # Note on shapes: NeetCode/PyTorch linear layers store weights as 
        # (out_features, in_features). 
        # Therefore, the matrix multiplication is: W * input + b
        # ==========================================
        
        # Layer 1: Linear transformation
        z1 = np.dot(W1, x) + b1       
        
        # Layer 1 Activation: ReLU (Rectified Linear Unit)
        # Sets any negative values in z1 to 0
        y1 = np.maximum(0, z1)        
        
        # Layer 2: Linear transformation (Output Layer)
        ypred = np.dot(W2, y1) + b2   
        
        # Loss Calculation: Mean Squared Error (MSE)
        mse = np.mean((ypred - y_true)**2)
        loss = np.round(mse, 4)
        
        # ==========================================
        # BACKWARD PASS (Backpropagation)
        # Applying the chain rule to calculate gradients from the output 
        # all the way back to the first layer's parameters.
        # ==========================================
        n = ypred.shape[0] # Number of output neurons
        
        # 1. Output Gradient (dL / d_ypred)
        # The derivative of the MSE loss function: (2/n) * (predictions - targets)
        dl = (2.0 / n) * (ypred - y_true)
        
        # 2. Layer 2 Gradients (dL/dW2 and dL/db2)
        # The gradient for the bias is directly the incoming gradient.
        db2 = dl
        # The gradient for the weights is the outer product of the incoming gradient 
        # and the layer's input. Order matters: (dl, y1) creates a matrix that 
        # correctly matches the shape of W2.
        dw2 = np.outer(dl, y1)        
        
        # 3. Hidden Layer Gradient (dL / d_y1)
        # To pass the gradient backward through Layer 2, we multiply the transpose 
        # of the weights by the incoming gradient.
        dl1 = np.dot(W2.T, dl)        
        
        # 4. ReLU Gradient (dL / d_z1)
        # The derivative of ReLU is 1 if z1 > 0, and 0 otherwise. 
        # We multiply the incoming gradient (dl1) element-wise by this boolean mask.
        dz = dl1 * (z1 > 0)           
        
        # 5. Layer 1 Gradients (dL/dW1 and dL/db1)
        db1 = dz
        # Again, taking the outer product to match the shape of W1.
        dw1 = np.outer(dz, x)         
        
        # ==========================================
        # RETURN FORMATTING
        # The environment expects standard Python lists rounded to 4 decimal places,
        # so we convert the NumPy arrays using np.round() and .tolist()
        # ==========================================
        return {
            'loss': float(loss),
            'dW1': np.round(dw1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dw2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
