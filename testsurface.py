#!/usr/bin/env python2

import sys
from pykrige.ok import OrdinaryKriging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata as gd

import pybrain.datasets as pd
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

top = np.array([
 [9682.281, 6378.745, 51.9687],
 [8145.462, 452.817, 24.57328],
 [10092.948, 986.992, 30.80156],
 [8667.571, 1742.01, 38.85036],
 [4689.053, 2337.65, 26.08584],
 [1797.474, 683.645, 39.68615],
 [9197.302, 4221.738, 41.2431 ],
 [420.679, 1522.198, 41.33592],
 [3202.198, 7808.856, 24.6531 ],
 [8994.722, 6092.849, 38.07091],
 [6961.339, 10426.288, 29.79357],
 [5033.938, 479.622, 27.37967],
 [4358.568, 2003.794, 27.81732],
 [5245.839, 900.012, 26.36676],
 [1544.185, 4767.227, 25.01236],
 [1922.522, 943.193, 26.45242],
 [1850.522, 4684.301, 25.57342]
 ])

bot = np.array([
 [9682.281, 6378.745, 51.9687],
 [8145.462, 452.817, 24.57328],
 [10092.948, 986.992, 30.80156],
 [8667.571, 1742.01, 38.85036],
 [4689.053, 2337.65, 26.08584],
 [1797.474, 683.645, 39.68615],
 [9197.302, 4221.738, 41.2431 ],
 [420.679, 1522.198, 41.33592],
 [3202.198, 7808.856, 24.6531 ],
 [8994.722, 6092.849, 38.07091],
 [6961.339, 10426.288, 29.79357],
 [5033.938, 479.622, 27.37967],
 [4358.568, 2003.794, 27.81732],
 [5245.839, 900.012, 26.36676],
 [1544.185, 4767.227, 25.01236],
 [1922.522, 943.193, 26.45242],
 [1850.522, 4684.301, 25.57342]
 ])


# 'nearest', 'linear', 'cubic'
interpolationmethod = 'cubic'
p = 2
extrapolation_interval = 30


def main():
    extrapolation_spots = get_plane(0, 10560, 0, 10600, extrapolation_interval)
    nearest_analysis(extrapolation_spots)
    kriging_analysis(extrapolation_spots)
    neural_analysis(extrapolation_spots)


def neural_analysis(extrapolation_spots):
    top_extra = neural_net(extrapolation_spots, top)
    gridx_top, gridy_top, gridz_top = interpolation(top_extra)
    plot(top, gridx_top, gridy_top, gridz_top, method='snaps', title='_top_neural_net')

    bot_extra = neural_net(extrapolation_spots, bot)
    gridx_bot, gridy_bot, gridz_bot = interpolation(bot_extra)
    plot(bot, gridx_bot, gridy_bot, gridz_bot, method='snaps', title='_bot_neural_net')

    plot(np.concatenate((top, bot)), [gridx_top, gridx_bot],
              [gridy_top, gridy_bot],
              [gridz_top, gridz_bot], method='snaps', title='_both_neural_net', both=True)


def neural_net(extrapolation_spots, data):
    net = buildNetwork(2, 10, 1)
    ds = pd.SupervisedDataSet(2, 1)
    for row in top:
        ds.addSample((row[0], row[1]), (row[2],))
    trainer = BackpropTrainer(net, ds)
    trainer.trainUntilConvergence()

    new_points = np.zeros((len(extrapolation_spots), 3))
    new_points[:, 0] = extrapolation_spots[:, 0]
    new_points[:, 1] = extrapolation_spots[:, 1]
    for i in range(len(extrapolation_spots)):
        new_points[i, 2] = net.activate(extrapolation_spots[i, :2])
    combined = np.concatenate((data, new_points))
    return combined


def nearest_analysis(extrapolation_spots):
    top_extra = extrapolation(top, extrapolation_spots, method='nearest')
    bot_extra = extrapolation(bot, extrapolation_spots, method='nearest')
    gridx_top, gridy_top, gridz_top = interpolation(top_extra)
    plot(top, gridx_top, gridy_top, gridz_top, method='snaps', title='_top_nearest')
    gridx_bot, gridy_bot, gridz_bot = interpolation(bot_extra)
    plot(bot, gridx_bot, gridy_bot, gridz_bot, method='snaps', title='_bot_nearest')

    plot(np.concatenate((top, bot)), [gridx_top, gridx_bot],
              [gridy_top, gridy_bot],
              [gridz_top, gridz_bot], method='snaps', title='_both_nearest', both=True)


def kriging_analysis(extrapolation_spots):
    top_extra = extrapolation(top, extrapolation_spots, method='kriging')
    bot_extra = extrapolation(bot, extrapolation_spots, method='kriging')
    gridx_top, gridy_top, gridz_top = interpolation(top_extra)
    plot(top, gridx_top, gridy_top, gridz_top, method='snaps', title='_top_kriging')
    gridx_bot, gridy_bot, gridz_bot = interpolation(bot_extra)
    plot(bot, gridx_bot, gridy_bot, gridz_bot, method='snaps', title='_bot_kriging')

    plot(np.concatenate((top, bot)), [gridx_top, gridx_bot],
              [gridy_top, gridy_bot],
              [gridz_top, gridz_bot], method='snaps', title='_both_kriging', both=True)


def nearest_neighbor_interpolation(data, x, y, p=0.5):
    """
    Nearest Neighbor Weighted Interpolation
    http://paulbourke.net/miscellaneous/interpolation/
    http://en.wikipedia.org/wiki/Inverse_distance_weighting

    :param data: numpy.ndarray
        [[float, float, float], ...]
    :param p: float=0.5
        importance of distant samples
    :return: interpolated data
    """
    n = len(data)
    vals = np.zeros((n, 2), dtype=np.float64)
    distance = lambda x1, x2, y1, y2: (x2 - x1)**2 + (y2 - y1)**2
    for i in range(n):
        vals[i, 0] = data[i, 2] / (distance(data[i, 0], x, data[i, 1], y))**p
        vals[i, 1] = 1          / (distance(data[i, 0], x, data[i, 1], y))**p
    z = np.sum(vals[:, 0]) / np.sum(vals[:, 1])
    return z


def get_plane(xl, xu, yl, yu, i):
    xx = np.arange(xl, xu, i)
    yy = np.arange(yl, yu, i)
    extrapolation_spots = np.zeros((len(xx) * len(yy), 2))
    count = 0
    for i in xx:
        for j in yy:
            extrapolation_spots[count, 0] = i
            extrapolation_spots[count, 1] = j
            count += 1
    return extrapolation_spots


def kriging(data, extrapolation_spots):
    """
    https://github.com/bsmurphy/PyKrige

    NOTE: THIS IS NOT MY CODE

    Implementing a kriging algorithm is out of the scope of this homework

    Using a library. See attached paper for kriging explanation.
    """
    gridx = np.arange(0.0, 200, 10)
    gridy = np.arange(0.0, 200, 10)
    # Create the ordinary kriging object. Required inputs are the X-coordinates of
    # the data points, the Y-coordinates of the data points, and the Z-values of the
    # data points. If no variogram model is specified, defaults to a linear variogram
    # model. If no variogram model parameters are specified, then the code automatically
    # calculates the parameters by fitting the variogram model to the binned
    # experimental semivariogram. The verbose kwarg controls code talk-back, and
    # the enable_plotting kwarg controls the display of the semivariogram.
    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='spherical',
                                 verbose=False, nlags=100)

    # Creates the kriged grid and the variance grid. Allows for kriging on a rectangular
    # grid of points, on a masked rectangular grid of points, or with arbitrary points.
    # (See OrdinaryKriging.__doc__ for more information.)
    z, ss = OK.execute('grid', gridx, gridy)
    return gridx, gridy, z, ss


def extrapolation(data, extrapolation_spots, method='nearest'):
    if method == 'kriging':
        xx, yy, zz, ss = kriging(data, extrapolation_spots)
        new_points = np.zeros((len(yy) * len(zz), 3))
        count = 0
        for i in range(len(xx)):
            for j in range(len(yy)):
                new_points[count, 0] = xx[i]
                new_points[count, 1] = yy[j]
                new_points[count, 2] = zz[i, j]
                count += 1
        combined = np.concatenate((data, new_points))
        return combined

    if method == 'nearest':
        new_points = np.zeros((len(extrapolation_spots), 3))
        new_points[:, 0] = extrapolation_spots[:, 0]
        new_points[:, 1] = extrapolation_spots[:, 1]
        for i in range(len(extrapolation_spots)):
            new_points[i, 2] = nearest_neighbor_interpolation(data,
                                    extrapolation_spots[i, 0], extrapolation_spots[i, 1], p=p)
        combined = np.concatenate((data, new_points))
        return combined


def interpolation(data):
    gridx, gridy = np.mgrid[0:10560:50j, 0:10600:50j]
    gridz = gd(data[:, :2],data[:, 2], (gridx, gridy), method=interpolationmethod)
    return gridx, gridy, gridz


def plot(data, gridx, gridy, gridz, method='rotate', title='nearest', both=False):
    def update(i):
        ax.view_init(azim=i)
        return ax,

    if method == 'rotate':
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

        ax.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')

        animation.FuncAnimation(fig, update, np.arange(360 * 5), interval=1)
        plt.show()

    elif method== 'snaps':
        fig = plt.figure(figsize=(10, 10))

        if both:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot_wireframe(gridx[0], gridy[0], gridz[0], alpha=0.5)
            ax.plot_wireframe(gridx[1], gridy[1], gridz[1], alpha=0.5)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')
        else:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')

        plt.savefig('snaps_{}.png'.format(title))

    elif method == 'contour':
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

        ax.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')

        ax.contourf(gridx, gridy, gridz, zdir='z', offset=np.min(data[:, 2]), cmap=cm.coolwarm)
        ax.contourf(gridx, gridy, gridz, zdir='x', offset=0, cmap=cm.coolwarm)
        ax.contourf(gridx, gridy, gridz, zdir='y', offset=0, cmap=cm.coolwarm)
        ax.view_init(azim=45)
        plt.show()


if __name__ == '__main__':
    sys.exit(main())
