"""
Date: 7-11-2023
Author: GTB

QEL to fit a continuos-variable 1D function and extremize it. The function is convex in the domain considered (only one maximum).

Steps
-----
1. Generate random uniformly distributed samples and labels
    1a. Split into a train and a test set
2. Build the QNN
    2a. Apply RY(2jarccos(x)) feature map. Params: x.s
    2b. Apply Hardware Efficient Ansatz (HEA). Params: \theta.
    2c. Calculate the expectation value <M>=<Z^{\otimes n}>.
3. Define loss function (MSE).
4. Minimise the loss function by updating \theta (ADAM, 50 epochs, lr=0.5).
    4a. Check convergence and test error
5. Fix \theta and allow the x to change
6. Maximise the trained model to find x_opt (ADAM, 50 epochs, lr=0.5)
    6a. Check convergence of extremal value and extremising parameters
    6b. Check maximisation trajectory
"""
import pennylane as qml
from pennylane import numpy as np


def train_test_split(X, y, n_test=1):
    test_idxs = np.random.choice(len(y), size=n_test, replace=False)
    train_idxs = np.setdiff1d(np.arange(len(y)), test_idxs)
    return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]


def U(x):
    for i in range(len(x)):
        qml.RY(2*i*np.arccos(x), wires=i)


def W(theta):  # may be different from what implemented in the paper
    n_qubits = theta.shape[1]
    qml.BasicEntanglerLayers(theta, wires=range(n_qubits), rotation=qml.RY)


def main():
    np.random.seed = 42

    M = 30
    X = np.random.uniform(low=0., high=1., size=M)
    y = np.sin(5*X)

    X_train, y_train, X_test, y_test = train_test_split(X, y, n_test=10)

    n_qubits = 3
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(device=dev, interface='autograd')
    def qnn(x, theta):
        U(x)
        W(theta)
        qml.expval(qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliX(2))


if __name__ == '__main__':
    main()
