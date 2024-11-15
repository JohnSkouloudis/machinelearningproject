import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionEP34:

    #2.1
    def __init__(self , lr=1e-2):
        self.w = None
        self.b = None
        self.lr = lr

        self.N = None
        self.p = None

        self.l_grad_w = None
        self.l_grad_b = None
        self.f = None
    
    #2.2
    def init_parameters(self):

        self.w = np.random.rand(self.p)*0.1
        self.b = np.random.rand()*0.1

    #2.3
    def forward(self,X):

        z = X @ self.w + self.b
        self.f = 1/(1+np.exp(-z))

    #return the probabilities of each class being 1 P(y=1|x)
    def predict_proba(self,X):

        z = X @ self.w + self.b
        probabilities = 1/(1+np.exp(-z))

        return probabilities    
    
    #2.4
    def predict(self,X):

        probabilities = self.predict_proba(X)

        return  (probabilities > 0.5).astype(int)

    #2.5
    def loss(self,X,y):
        
        loss = -1/self.N * np.sum( y*np.log(self.predict_proba(X)) + (1-y)*np.log(1-self.predict_proba(X)) )
        return loss

    #2.6
    def backward(self,X,y):  

       self.l_grad_w = -1/self.N * ( X.T @ (y -self.f) )

       self.l_grad_b = -1/self.N * np.sum(y - self.f)

        
    
    #2.7
    def step(self):

        self.w -= self.lr * self.l_grad_w
    
        self.b -= self.lr * self.l_grad_b
    
    #2.8
    def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000, show_line=False):
        #check if X,y are numpy arrays and the dimensions are compatible
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
        if (X.shape[0] != y.shape[0]) and y.shape[1] != 1:
            raise ValueError("The number of samples in X and y must be equal and y must be a column vector")
        
        self.N, self.p = X.shape
        self.init_parameters()
        
        # change the sample order 
        order = np.arange(self.N)
        np.random.shuffle(order)
        X = X[order]
        y = y[order]
        
        for iteration in range(iterations):
            #get the next batch
            if batch_size is None:
                X_batch = X
                y_batch = y
            else:
                start = (iteration * batch_size) % self.N
                finish = start + batch_size
                if finish > self.N:
                    X_batch = np.concatenate((X[start:], X[:finish % self.N]), axis=0)
                    y_batch = np.concatenate((y[start:], y[:finish % self.N]), axis=0)
                else:
                    X_batch = X[start:finish]
                    y_batch = y[start:finish]
            
            
            self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.step()
            
            #show the loss and the iteration requested
            if (iteration + 1) % show_step ==0 :
                current_loss = self.loss(X, y)
                print(f"Iteration {iteration + 1}: Loss = {current_loss}")

        #show the diagram       
        if show_line:
                    self.show_line(X, y)
    
    #2.9
    def show_line(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot a binary problem and a line. This function assumes 2-D problem
        (just plots the first two dimensions of the data)
        """
        if (X.shape[1] != 2):
            print("Not plotting: Data is not 2-dimensional")
            return

        idx0 = (y == 0)
        idx1 = (y == 1)
        X0 = X[idx0, :2]
        X1 = X[idx1, :2]
        plt.plot(X0[:, 0], X0[:, 1], 'gx')
        plt.plot(X1[:, 0], X1[:, 1], 'ro')
        min_x = np.min(X, axis=0)
        max_x = np.max(X, axis=0)
        xline = np.arange(min_x[0], max_x[0], (max_x[0] - min_x[0]) / 100)
        yline = (self.w[0]*xline + self.b) / (-self.w[1])
        plt.plot(xline, yline, 'b')
        plt.show()




