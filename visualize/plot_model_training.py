import pickle
import sys

from matplotlib import animation, pyplot as plt
import numpy as np

from project_directories import PICKLE_PATH, FIGURES_PATH


N_EPOCHS = 100


def main():
    with open(PICKLE_PATH / 'X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(PICKLE_PATH / 'y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(PICKLE_PATH / 'x_grid.pkl', 'rb') as f:
        x_grid = pickle.load(f)
    with open(PICKLE_PATH / 'model_preds_training.pkl', 'rb') as f:
        model_preds = pickle.load(f)

    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, edgecolor='k', facecolor='w')
    ax.plot(x_grid, np.sin(5*x_grid), color='k',
            linewidth=1, label='ground truth')
    ax.plot(x_grid, model_preds[0], 'g--', linewidth=1.5, label='initial')[0]
    line1 = ax.plot(x_grid, model_preds[0],
                    color='g', linewidth=1.5)[0]
    ax.set_xlim((x_grid[0], x_grid[-1]))
    ax.set_ylim([-1., 1.])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("QNN modeling of f(x)=sin(5x) through training")
    ax.grid()
    ax.legend()

    def animation_step(frame):
        line1.set_xdata(x_grid)
        line1.set_ydata(model_preds[frame])
        return line1

    fig.tight_layout()
    ani = animation.FuncAnimation(
        fig=fig, func=animation_step, frames=N_EPOCHS, interval=50)
    ani.save(FIGURES_PATH / "qel_sin5x_model_during_training.mp4",
             writer='ffmpeg')
    plt.close(fig)


if __name__ == '__main__':
    main()
