from pathlib import Path
import pickle
from matplotlib import animation, pyplot as plt

from project_directories import PICKLE_PATH, FIGURES_PATH


def main():
    with open(PICKLE_PATH / 'epochs.pkl', 'rb') as f:
        epochs = pickle.load(f)
    with open(PICKLE_PATH / 'cost_mse.pkl', 'rb') as f:
        training_losses = pickle.load(f)

    fig, ax = plt.subplots()
    line = ax.plot(epochs[0], training_losses[0],
                   color='k', linewidth=1.5)[0]
    ax.set_xlim([0, len(epochs)])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title("QNN learning of f(x)=sin(5x)")
    ax.grid()

    def animation_step(frame):
        line.set_xdata(epochs[:frame])
        line.set_ydata(training_losses[:frame])
        return line

    fig.tight_layout()
    ani = animation.FuncAnimation(
        fig=fig, func=animation_step, frames=len(epochs), interval=50)
    ani.save(FIGURES_PATH/"qel_sin5x_training_curve.mp4",
             writer='ffmpeg')
    plt.close(fig)


if __name__ == '__main__':
    main()
