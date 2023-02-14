import matplotlib.pyplot as plt
import numpy as np

def set_viz_properties():
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    
    return None

def save_fig(IMAGES_PATH, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_decision_boundary(clf, X, y, cmap, axes=[-1.5, 2.4, -1, 1.5], alpha=1.0):
    # axes=[-1.5, 2.4, -1, 1.5]
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)
    colors = {"Wistia": ["#78785c", "#c47b27"], "Pastel1": ["red", "blue"]}
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[cmap][idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)

def plot_regression_predictions(tree_reg, X, y, axes=[-0.5, 0.5, -0.05, 0.25]):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$")
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")
    
# def plot_decision_boundary(clf, X, y, alpha=1.0):
    # """
    # from 07_ensemble_learning_and_random_forests
    # """
#     axes=[-1.5, 2.4, -1, 1.5]
#     x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
#                          np.linspace(axes[2], axes[3], 100))
#     X_new = np.c_[x1.ravel(), x2.ravel()]
#     y_pred = clf.predict(X_new).reshape(x1.shape)
    
#     plt.contourf(x1, x2, y_pred, alpha=0.3 * alpha, cmap='Wistia')
#     plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8 * alpha)
#     colors = ["#78785c", "#c47b27"]
#     markers = ("o", "^")
#     for idx in (0, 1):
#         plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
#                  color=colors[idx], marker=markers[idx], linestyle="none")
#     plt.axis(axes)
#     plt.xlabel(r"$x_1$")
#     plt.ylabel(r"$x_2$", rotation=0)