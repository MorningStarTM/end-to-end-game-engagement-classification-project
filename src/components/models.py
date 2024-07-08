import os
import torch.nn as nn
import torch.optim as optim
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, out_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)

class EngageModel:
    """
        Model class

        Args:
            input_dim (int) : input data dimension
            output_dim (int) : output data dimension
            learning_rate (float) 
    """
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.best_accuracy = 0.0

    def train(self, train_loader, val_loader, epochs=10, save_path='best_model.pth'):
        """
        this function for model training.

        Args:
            train loader (Data loader)
            val_loader (Data loader)
            epochs (int)
            save path (str)
        """
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                train_accuracy = self.evaluate(train_loader)
                val_accuracy = self.evaluate(val_loader)

                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    self.save_model(save_path)
                    print(f'Best model saved with accuracy: {val_accuracy * 100:.2f}%')

            self.model.train()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy * 100:.2f}%, Val Accuracy: {val_accuracy * 100:.2f}%')

    def predict(self, X):
        """
        Function for prediction

        Args:
            X : data
        
        Return:
            predicted label
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted
    
    def evaluate(self, data_loader):
        """
        This function for model evaluation

        Args:
            data loader (Data loader)

        Return:
            accuracy (float)
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        accuracy = correct / total
        return accuracy

    def save_model(self, path):
        """
        Function for save model

        Args:
            Path (str): location for save the model
        """
        torch.save(self.model.state_dict(), os.path.join('artifacts',path))
        print(f"model saved at {path}")

    def load_model(self, path):
        """
        Function for load the model

        Args:
            path (str) : location of saved model
        """
        self.model.load_state_dict(torch.load(os.path.join('artifacts',path)))
        self.model.to(self.device)
        print("Model loaded")
