import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def fit_2gauss(X, Y):
    def two_gaussians(x, a1, mu1, sigma1, a2, mu2, sigma2):
        gauss1 = a1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2))
        gauss2 = a2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
        return gauss1 + gauss2

    # Set lower bounds for all parameters to 0
    bounds = ([0, -1000, 0, 0, -1000, 0], [np.inf, 1000, np.inf, np.inf, 1000, np.inf])
    # Initial guess parameters for Gaussian fit
    peak_X = X[np.argmax(Y)]
    initial_guess = [0.0107, peak_X, 0.101, 0.00349, peak_X + 5, 0.21]
    # Fit the data
    popt, pcov = curve_fit(two_gaussians, X, Y, p0=initial_guess)
    # The fitted parameters are in popt
    print('Fitted parameters:', popt)

    # Calculate goodness of fit: sum of the residuals squared
    residuals = Y - two_gaussians(X, *popt)
    goodness = np.sum(residuals ** 2)
    print('Goodness of fit:', goodness)

    return popt, goodness


def calculate_y(X_seq, cf2):
    # Unpack the coefficients
    amp1, mu1, sigma1, amp2, mu2, sigma2 = cf2

    # Calculate the sum of two Gaussian functions
    return amp1 * np.exp(-(X_seq - mu1) ** 2 / (2 * sigma1 ** 2)) + amp2 * np.exp(
        -(X_seq - mu2) ** 2 / (2 * sigma2 ** 2))


def intersect_gaussians(mu1, mu2, sigma1, sigma2):
    # Check if the two Gaussians are identical
    if mu1 == mu2 and sigma1 == sigma2:
        return np.nan, np.nan

    # Check if the two Gaussians have the same standard deviation
    if sigma1 == sigma2:
        return (mu1 + mu2) / 2, math.inf * (mu2 - mu1)

    # Compute solutions when the standard deviations are different
    SIGMA1 = sigma1 ** 2
    SIGMA2 = sigma2 ** 2

    aux1 = mu2 * SIGMA1 - mu1 * SIGMA2
    aux2 = sigma1 * sigma2 * np.sqrt((mu1 - mu2) ** 2 + 2 * (SIGMA2 - SIGMA1) * np.log(sigma2 / sigma1))
    aux3 = 1 / (SIGMA1 - SIGMA2)

    # Compute intersection points
    x1 = (aux1 - aux2) * aux3
    x2 = (aux1 + aux2) * aux3

    # Sort solutions from smaller to larger
    return min(x1, x2), max(x1, x2)


def threshold_estimator(filtered_data):
    # Assuming filtered_data and timestamps are defined
    signal_values = filtered_data.dropna().values
    signal_values = signal_values[signal_values > 0]
    signal_values = np.log(signal_values)
    # Creating histogram
    Y, X = np.histogram(signal_values, bins=1000)
    X = (X[1:] + X[:-1]) / 2  # Get bin centers
    Y = Y / np.sum(Y)  # Normalize histogram

    # fit data with 2 gaussians
    params_optimized, goodness = fit_2gauss(X, Y)
    a1, mu1, sigma1, a2, mu2, sigma2 = params_optimized
    intersections = np.array(intersect_gaussians(mu1, mu2, sigma1, sigma2))
    log_threshold = intersections[(intersections > mu1) & (intersections < mu2)]

    return log_threshold, X, Y, params_optimized


def show_threshold_estimation(X, Y, params_optimized, threshold=None):
    plt.figure()
    plt.plot(X, Y, linewidth=4)

    X_seq = np.linspace(min(X), max(X), 500)
    Y_seq = calculate_y(X_seq, params_optimized)  # This function should be implemented according to your fit function
    plt.plot(X_seq, Y_seq, color='red', linewidth=1)
    if threshold:
        plt.axvline(x=threshold, linewidth=2, color='red')
    plt.show()
