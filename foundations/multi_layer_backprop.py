import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2= np.array(b2)
        y_true= np.array(y_true)
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        #Forward pass
        z1=np.dot(W1,x)+b1
        y1=np.maximum(0,z1)
        ypred=np.dot(W2,y1)+b2
         #Loss cal
        mse=np.mean((ypred-y_true)**2)
        loss=np.round(mse,4)
        #backward pass
        n=ypred.shape[0]
         #op gradient
        dl=(2.0/n)*(ypred-y_true)
        #2 layer2 gradient
        db2=dl
        # FLIPPED: dl first, y1 second. This creates a (1, 2) matrix instead of (2, 1)
        dw2 = np.outer(dl, y1)
        #hd layer gd
        dl1=np.dot(dl,W2)
        dz=dl1*(z1>0)
        #layer1 gd
        db1=dz
        dw1=np.outer(db1,x)

        return {
            'loss': float(loss),
            'dW1': np.round(dw1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dw2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
        pass
