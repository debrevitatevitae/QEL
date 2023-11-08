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
import sys
from matplotlib import pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split


def mse(preds, labels):
    n = len(labels)
    loss = 0.
    for pp, ll in zip(preds, labels):
        loss += (pp-ll)**2
    return 1/n*loss


def main():
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

    x_grid = np.linspace(x_min, x_max, 100, requires_grad=False)
    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, edgecolor='k', facecolor='w')
    ax.plot(x_grid, np.sin(5*x_grid), color='k',
            linewidth=1, label='ground truth')
    ax.plot(x_grid, qnn(x_grid, theta), color='b',
            linewidth=1.5, label='initial')

    opt = qml.AdamOptimizer(stepsize=0.5)
    n_epochs = 100

    for ep in range(n_epochs):
        theta, _, _ = opt.step(cost_step_1, theta, X_train, y_train)

        # print(
        #     f"Epoch: {ep:3d} | Train loss: {cost_step_1(theta, X_train, y_train):.4f}")

    print(f"Test MSE = {cost_step_1(theta, X_test, y_test)}")

    ax.plot(x_grid, qnn(x_grid, theta), color='g',
            linewidth=1.5, label='final')
    plt.show()

    # QEL step 2: find the extremizing value of the learned model
    theta_opt = theta

    def cost_step_2(x):
        return -qnn(x, theta_opt)[0]

    opt = qml.AdamOptimizer(stepsize=0.5)
    n_epochs = 50
    x = np.random.uniform(x_min, x_max, size=(1,), requires_grad=True)

    for ep in range(n_epochs):
        x = opt.step(cost_step_2, x)

        print(
            f"Epoch: {ep:3d} | Current extremiser: {x[0]:.4f} | Model value: {-cost_step_2(x):.4f}")


if __name__ == '__main__':
    main()
