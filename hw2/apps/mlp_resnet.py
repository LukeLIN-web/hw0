import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, dim),
            norm(dim),
        )
    model = nn.Residual(fn)
    model = nn.Sequential(model, nn.ReLU())
    return model
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    models = [nn.Linear(dim, hidden_dim),nn.ReLU()]
    for i in range(num_blocks):
        models.append(ResidualBlock(hidden_dim, hidden_dim, norm, drop_prob))
    models.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*models)
    ### END YOUR SOLUTION



def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    for X, y in dataloader:
        logits = model.forward(X)
        loss = nn.SoftmaxLoss().forward(logits, y)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    data = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    train_loader = ndl.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    epoch(train_loader, MLPResNet(784, hidden_dim), optimizer(0.001, 0.001))
    return epoch(test_loader, MLPResNet(784, hidden_dim))
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
