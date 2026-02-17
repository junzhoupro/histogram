import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,  # require LaTeX and type1cm on Ubuntu
        "font.family": "serif",
        # "text.latex.preamble": [
        #     r"""
        # \usepackage{libertine}
        # \usepackage[libertine]{newtxmath}
        # """
        # ],
    }
)


def plot_rgb_histogram(fp: str, fg_hist: np.ndarray, pred_hist: np.ndarray, gt_hist: np.ndarray) -> None:
    """Visualize and compare input histograms of shape (batch_size, 3, num_bins)."""
    N, C, B = gt_hist.shape
    fig, axes = plt.subplots(nrows=C, ncols=N, figsize=(3 * N + 2, 3 * C))

    bins = np.linspace(0, B - 1, num=B)
    for c, c_color in enumerate(("red", "green", "blue")):
        for n in range(N):
            fg = axes[c, n].plot(bins, fg_hist[n, c])
            # print(f"n: {n} c: {c}")
            # print(f"fg_hist[n, c]: {fg_hist[n, c]}")
            pred = axes[c, n].plot(bins, pred_hist[n, c])
            gt = axes[c, n].plot(bins, gt_hist[n, c])
            axes[c, n].margins(x=0, y=0)
            axes[c, n].grid(linestyle="--")
        axes[c, 0].set_ylabel(c_color)

    fig.legend([fg, pred, gt], labels=["input", "output", "gt"], loc="upper right")

    # adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.93, 1.01])
    plt.savefig(fp, dpi=300)
    plt.close()


def plot_single_rgb_histogram(fp: str, fg_hist: np.ndarray, bg_hist: np.ndarray, pred_hist: np.ndarray) -> None:
    """Visualize and compare input histograms of shape (1, 3, num_bins)."""
    N, C, B = pred_hist.shape
    assert N == 1, "this function should not be called when the input batch size is 1"
    fig, axes = plt.subplots(nrows=1, ncols=C, figsize=(3 * C, 3 * N))
    bins = np.linspace(0, B - 1, num=B)

    axes[0].plot(bins, fg_hist[0, 0], color="red")
    axes[0].plot(bins, fg_hist[0, 1], color="green")
    axes[0].plot(bins, fg_hist[0, 2], color="blue")
    axes[0].set_title("Foreground")

    axes[1].plot(bins, bg_hist[0, 0], color="red")
    axes[1].plot(bins, bg_hist[0, 1], color="green")
    axes[1].plot(bins, bg_hist[0, 2], color="blue")
    axes[1].set_title("Background")

    axes[2].plot(bins, pred_hist[0, 0], color="red")
    axes[2].plot(bins, pred_hist[0, 1], color="green")
    axes[2].plot(bins, pred_hist[0, 2], color="blue")
    axes[2].set_title("Predicted")

    for ax in axes:
        ax.margins(x=0, y=0)
        ax.grid(linestyle="--")

    # adjust layout to make room for the legend
    plt.tight_layout()
    plt.savefig(fp, dpi=300)
    plt.close()
