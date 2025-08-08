"""
Hypnogram functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

stages_names = {0: 'WAKE', 1: 'IS/QW', 2: 'NREM', 3: 'SWS', 4: 'REM'}
COLOR_SLEEP_STAGES = {
    'WAKE': (0.5, 0.2, 0.1),
    'IS/QW': (0.5, 0.3, 1),
    'NREM': (1, 0.5, 1),
    'SWS': (0.8, 0, 0.7),
    'REM': (0.1, 0.7, 0)
}


def plot_hypnogram(
    hypnogram,
    x_hypno,
    axe_plot=None,
    title="Hypnogram",
):
    """
    Plot a hypnogram.

    :param hypnogram: int[] or numpy.array
        hypnogram, list of sleep stages
    :param x_hypno: int[] or numpy.array
        list of timestamps
    :param axe_plot: axes matplotlib
        axes where to plot the graph

    :return: ax: axes matplotlib
        axe of the hypnogram graph
    """

    ytick_substage = [4, 2, 1.5, 1, 3]  # position of the stages in the graph, down to up
    ylabel_substage = ['SWS', 'NREM', 'IS/QW', 'REM', 'WAKE']  # stages in the graph, down to up
    graph_hypno = np.asarray([ytick_substage[int(stage)] for stage in hypnogram])

    # plot
    if axe_plot is None:
        fig, axs = plt.subplots(1, 1, figsize=(40, 25))
        ax = np.ravel(axs)[0]
        plt.rcParams.update({'font.size': 40})
    else:
        ax = axe_plot

    ax.step(x_hypno, graph_hypno, 'k', linewidth=0.5)

    ax.set_yticks(np.sort(ytick_substage))
    ax.set_yticklabels(ylabel_substage)
    ax.set_ylim(0.5, 4.5)
    ax.set_xlim(min(x_hypno), max(x_hypno) + 0.1)

    if axe_plot is None:
        fig.show()

    return ax


def compute_sleep_stage_metrics(hypnogram, epoch_duration):
    """
    Sleep metrics from hypnogram
    :param hypnogram: 1D numpy.array
        list of sleep stages
    :return:
    stage_stat: dict
        Stage duration and Sleep Stage ratio
    """
    hypnogram = np.asarray(hypnogram)
    total_time = len(hypnogram) * epoch_duration

    stage_duration = {}
    stage_percentage = {}
    # sleep architecture - sleep stage %
    for stage_idx, stage_name in stages_names.items():
        duration = sum(hypnogram == stage_idx) * epoch_duration
        stage_duration[stage_name] = duration
        stage_percentage[stage_name] = duration / total_time

    return stage_duration, stage_percentage


def compute_stage_transitions(hypnogram, epoch_duration):
    """
    Compute stage durations and transition matrices
    :param hypnogram: 1D.numpy.array or int[]
    :return:
    stage_durations - 1D.numpy.array
    transition_matrix_2d - 2D.numpy.array
    """
    hypnogram = np.asarray(hypnogram).astype(int)
    nb_stages = len(stages_names)

    # outputs
    stage_duration = {}
    transition_matrix_2d = np.zeros((nb_stages, nb_stages))

    # duration
    for stage_idx, stage_name in stages_names.items():
        if stage_idx >= 0:
            stage_duration[stage_name] = sum(hypnogram == stage_idx) * epoch_duration

    # duets
    for i, _ in enumerate(hypnogram[:-1]):
        transition_matrix_2d[hypnogram[i], hypnogram[i + 1]] += 1
    np.fill_diagonal(transition_matrix_2d, 0)

    return stage_duration, transition_matrix_2d


def transition_matrix_to_dict(transition_matrix):

    positive_idx = np.asarray(np.nonzero(transition_matrix)).T
    result = {}
    for idx in positive_idx:
        name = '-'.join(stages_names[i] for i in idx)
        result[name] = int(transition_matrix[tuple(idx)])

    return result


def plot_sleep_stage_transitions(
        hypnogram,
        epoch_duration,
        axe_plot=None,
        title="Transitions"
):
    """
    Plot a graph showing transitions between sleep stages
    :param hypnogram: 1D.numpy.array or int[]
    :param axe_plot: axes matplotlib - axes where to plot the graph
    :param title: str - title of the figure
    :return: ax: axes matplotlib
        axe of the graph
    """
    if hypnogram is None or len(hypnogram) == 0:
        return axe_plot

    # compute durations and transitions
    stage_duration, transition_matrix_2d = compute_stage_transitions(hypnogram, epoch_duration)
    stage_transitions = transition_matrix_to_dict(transition_matrix_2d)

    # plot
    ax = plot_graph_transition(stage_duration,
                               stage_transitions,
                               axe_plot=axe_plot,
                               title=title)
    return ax


def plot_graph_transition(
        stage_duration,
        stage_transitions,
        axe_plot=None,
        title="Transitions",
):
    """
    Plot a graph showing transitions betwen sleep stages
    :param stage_durations: dict - sleep stages durations (e.g {'N1': 1260, 'REM': 1470})
    :param stage_transitions: dict - number of (duet) transitions (e.g {'N1-N2': 8, 'REM-WAKE': 11})
    :param axe_plot: axes matplotlib - axes where to plot the graph
    :param title: str - title of the figure
    :return: ax: axes matplotlib
        axe of the hypnogram graph
    """
    # axe
    if axe_plot is None:
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        ax = np.ravel(axs)[0]
        plt.rcParams.update({'font.size': 40})
    else:
        ax = axe_plot
    ax.set_title(title)

    # Parameters
    STAGES_COORDINATES = {
        "WAKE": np.array([-5, 0]),
        "IS/QW": np.array([-1, 5]),
        "NREM": np.array([5, 0]),
        "SWS": np.array([5, 6]),
        "REM": np.array([-1, -5]),
    }
    stages_dict = {v: k for k, v in stages_names.items()}

    #normalize stage transitions
    factor = np.sqrt(sum(stage_duration.values())) / 20
    sum_transitions = sum(stage_transitions.values())
    norm_transitions = {key: factor * value / sum_transitions for key, value in stage_transitions.items()}

    # network
    transition_graph = nx.MultiDiGraph()
    for stage_name, stage_idx in stages_dict.items():
        transition_graph.add_node(stage_name)
    for transition, weight in norm_transitions.items():
        stage_1, stage_2 = transition.split('-')
        transition_graph.add_edge(stage_1, stage_2,
                                  color=COLOR_SLEEP_STAGES[stage_1],
                                  weight=weight,
                                  label=str(weight))
    colors = nx.get_edge_attributes(transition_graph, 'color').values()
    weights = np.asarray(list(nx.get_edge_attributes(transition_graph, 'weight').values()))
    weights = np.sqrt(weights)
    durations = np.asarray(list(stage_duration.values()))
    durations = 3000 * durations / np.sum(durations)

    # Plot
    nx.draw(transition_graph, STAGES_COORDINATES,
            nodelist=list(stages_dict.keys()),
            node_size=durations,
            node_color=[COLOR_SLEEP_STAGES[k] for k in stages_dict.keys()],
            labels={k: k for k in stages_dict.keys()},
            edge_color=colors,
            width=list(weights),
            connectionstyle='arc3, rad = 0.1',
            ax=axe_plot
            )
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)

    if axe_plot is None:
        fig.show()

    return ax


def plot_stage_duration(
        stage_duration,
        axe_plot=None,
        unit="min",
        title=""):
    """
    Bar plot of sleep stage duration (in min)

    :param stage_duration: dict
    :param axe_plot: axes matplotlib
        axes where to plot the graph
    :param title: str

    :return: ax: axes matplotlib
        axe of the graph
    """

    # plot
    if axe_plot is None:
        fig, axs = plt.subplots(1, 1, figsize=(40, 25))
        ax = np.ravel(axs)[0]
        plt.rcParams.update({'font.size': 40})
    else:
        ax = axe_plot
    ax.set_title(title)

    #scale
    if unit == "min":
        stage_duration = {k: v / 60 for k, v in stage_duration.items()}
    elif unit == "h":
        stage_duration = {k: v / 3600 for k, v in stage_duration.items()}

    colors = [COLOR_SLEEP_STAGES[k] for k in stage_duration.keys()]
    ax.bar(list(stage_duration.keys()), list(stage_duration.values()), color=colors)
    ax.set_ylabel(f"duration ({unit})")

    if axe_plot is None:
        fig.show()

    return ax


def plot_stage_percentage(
        stage_percentage,
        axe_plot=None,
        title=""):
    """
    Pie plot of sleep stage percentage

    :param stage_percentage: dict
    :param axe_plot: axes matplotlib
        axes where to plot the graph
    :param title: str

    :return: ax: axes matplotlib
        axe of the graph
    """

    # plot
    if axe_plot is None:
        fig, axs = plt.subplots(1, 1, figsize=(40, 25))
        ax = np.ravel(axs)[0]
        plt.rcParams.update({'font.size': 40})
    else:
        ax = axe_plot
    ax.set_title(title)

    # colors
    colors = [COLOR_SLEEP_STAGES[k] for k in stage_percentage.keys()]

    if np.nan not in list(stage_percentage.values()):
        ax.pie(list(stage_percentage.values()),
               labels=list(stage_percentage.keys()),
               colors=colors,
               normalize=True,
               autopct=lambda x: str(round(x, 1)) + '%',
               )
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(-1, 1)

    if axe_plot is None:
        fig.show()

    return ax
