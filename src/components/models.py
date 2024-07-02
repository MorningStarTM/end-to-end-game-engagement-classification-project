import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class NeuralNetwork(nn.Module):
    """
    Model class 
    """
    def __init__(self, input_dim, out_dim):

        """
        Args:
            input dim (int) : input data dimension
            out_dim (int) : output data dimension
            hidden_dim (int) : hidden dimension
        
       """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    


class EngageModel:
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        """
        Model class

        Args:
            input_dim (int) : input data dimension
            output_dim (int) : output data dimension
            learning_rate (float) 
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Function for training the model.
        
        Args:
            X_train : training data
            y_train : taining lables
            epochs (int)
            batch_size (int)
        """
        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(X_train.size()[0])
            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


    def evaluate(self, X_test, y_test):
        """
        Function for model evaluation

        Args:
            X_test : testing data
            y_test : testing lables

        Return
            Accuracy (float)
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test).sum().item()
            accuracy = correct / y_test.size(0)
        return accuracy

    def predict(self, X):
        """
        Function for prediction

        Args:
            X : data
        
        Return:
            predicted lable

        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def save_model(self, path):
        """
        Function for save model

        Args:
            Path (str): location for save the model

        
        """
        torch.save(self.model.state_dict(), path)
        print(f"model saved at {path}")


    def load_model(self, path):
        """
        Function for load the model

        Args:
            path (str) : location of saved model
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print("Model loaded")

        
