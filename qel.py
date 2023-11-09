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
from pathlib import Path
import pickle

from matplotlib import animation, pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split

from project_directories import PICKLE_PATH
OUTPUT_PATH = Path() / "output"


def mse(preds, labels):

    n = len(labels)
    loss = 0.
    for pp, ll in zip(preds, labels):
        loss += (pp-ll)**2
    return 1/n*loss


def main():
    SAVE_COST_TRAINING = False
    SAVE_MODEL_TRAINING = False
    SAVE_EXTREMAL_FINDING = False

    np.random.seed(42)

    n_samples = 30
    x_min = 0.
    x_max = 1.
    X = np.random.uniform(low=x_min, high=x_max, size=n_samples)
    # X = np.linspace(x_min, x_max, n_samples)
    y = np.sin(5*X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    n_qubits = 3

    def U(x):
        for i in range(n_qubits):
            qml.RY(2*i*np.arccos(x), wires=i)

    def layer(theta_i):
        for j in range(n_qubits):
            qml.Rot(theta_i[j, 0], theta_i[j, 1], theta_i[j, 2], wires=j)
        qml.broadcast(unitary=qml.CNOT, pattern="ring",
                      wires=range(n_qubits))

    n_layers = 3

    def W(theta):  # may be different from what implemented in the paper
        for theta_i in theta:
            layer(theta_i)

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(device=dev, interface="autograd")
    def qnn(x, theta):
        U(x)
        W(theta)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    def cost_step_1(theta, x, y):
        preds = qnn(x, theta)
        return mse(preds, y)

    # QEL step 1: train the quantum model to reproduce the target function
    theta = .01 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)
    # qml.draw_mpl(qnn)(X_train[0], theta)

    x_grid = np.linspace(x_min, x_max, 50, requires_grad=False)

    opt = qml.AdamOptimizer(stepsize=0.5)
    n_epochs = 100
    training_losses = [cost_step_1(theta, X_train, y_train)]
    model_preds = [qnn(x_grid, theta)]

    for ep in range(n_epochs-1):
        theta, _, _ = opt.step(cost_step_1, theta, X_train, y_train)
        training_losses.append(cost_step_1(theta, X_train, y_train))
        model_preds.append(qnn(x_grid, theta))

        print(
            f"Epoch: {ep:3d} | Train loss: {cost_step_1(theta, X_train, y_train):.4f}")

    print(f"Test MSE = {cost_step_1(theta, X_test, y_test)}")

    if SAVE_COST_TRAINING:
        epochs = list(range(n_epochs))
        with open(PICKLE_PATH / 'epochs.pkl', 'wb') as f:
            pickle.dump(epochs, f)
        with open(PICKLE_PATH / 'cost_mse.pkl', 'wb') as f:
            pickle.dump(training_losses, f)

    if SAVE_MODEL_TRAINING:
        with open(PICKLE_PATH / 'X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open(PICKLE_PATH / 'y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open(PICKLE_PATH / 'x_grid.pkl', 'wb') as f:
            pickle.dump(x_grid, f)
        with open(PICKLE_PATH / 'model_preds_training.pkl', 'wb') as f:
            pickle.dump(model_preds, f)

    # QEL step 2: find the extremizing value of the learned model
    theta_opt = theta

    def cost_step_2(x):
        return -qnn(x, theta_opt)[0]

    opt = qml.AdamOptimizer(stepsize=0.5)
    n_epochs = 50
    x = np.random.uniform(x_min, x_max, size=(
        1,), requires_grad=True)  # initial guess

    x_opt_iters = [x[0]]
    f_opt_iters = [qnn(x, theta_opt)[0]]

    for ep in range(n_epochs-1):
        x = opt.step(cost_step_2, x)
        x_opt_iters.append(x[0])
        f_opt_iters.append(qnn(x, theta_opt)[0])

        print(
            f"Epoch: {ep:3d} | Current extremiser: {x[0]:.4f} | Model value: {-cost_step_2(x):.4f}")

    if SAVE_EXTREMAL_FINDING:
        with open(PICKLE_PATH / 'qnn_on_grid.pkl', 'wb') as f:
            pickle.dump(qnn(x_grid, theta_opt), f)
        with open(PICKLE_PATH / 'extremizer_iters.pkl', 'wb') as f:
            pickle.dump(x_opt_iters, f)
        with open(PICKLE_PATH / 'extreme_val_iters.pkl', 'wb') as f:
            pickle.dump(f_opt_iters, f)


if __name__ == '__main__':
    main()
