import pickle

from matplotlib import animation, pyplot as plt
import numpy as np
from project_directories import PICKLE_PATH, FIGURES_PATH


def main():
    with open(PICKLE_PATH / 'x_grid.pkl', 'rb') as f:
        x_grid = pickle.load(f)
    with open(PICKLE_PATH / 'qnn_on_grid.pkl', 'rb') as f:
        qnn_on_grid = pickle.load(f)
    with open(PICKLE_PATH / 'extremizer_iters.pkl', 'rb') as f:
        x_opt_iters = pickle.load(f)
    with open(PICKLE_PATH / 'extreme_val_iters.pkl', 'rb') as f:
        f_opt_iters = pickle.load(f)

    x_opt = np.pi / 10
    f_opt = 0.95706

    fig, ax = plt.subplots()
    ax.plot(x_grid, qnn_on_grid,
            color='g', linewidth=1.5)
    ax.scatter(x_opt, f_opt, marker='*',
               facecolor='w', edgecolor='b', label='max')
    line1 = ax.plot(x_opt_iters[0], f_opt_iters[0],
                    color='r', label='pred. of max')[0]
    ax.set_xlim([x_grid[0], x_grid[-1]])
    ax.set_ylim([-1., 1.])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Optimization trajectory to the maximizer")
    ax.grid()
    ax.legend()

    def animation_step(frame):
        line1.set_xdata(x_opt_iters[:frame])
        line1.set_ydata(f_opt_iters[:frame])
        return line1

    fig.tight_layout()
    ani = animation.FuncAnimation(
        fig=fig, func=animation_step, frames=len(f_opt_iters), interval=100)
    ani.save(FIGURES_PATH / "qel_sin5x_max_trajectory.mp4",
             writer='ffmpeg')
    plt.close(fig)


if __name__ == '__main__':
    main()
