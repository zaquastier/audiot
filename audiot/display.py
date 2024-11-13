import matplotlib.pyplot as plt
import ipywidgets as widgets
import IPython

from .utils import *

def plot_signals(support, signals, labels=None, title=None, colors=None, figsize=[10, 4]):
    """
    Plots multiple signals with corresponding labels on the same frequency spectrum.

    Args:
        support (np.ndarray): Frequency support (in Hz).
        signals (arr[np.ndarray]): Signals to plot.
        labels (arr[str]): Labels for each signal.
        title (str): Title of the plot.
    """

    plt.figure(figsize=figsize)

    # Plot each signal with its label and color if provided
    for i, signal in enumerate(signals):
        label = labels[i] if i < len(labels) else f"Signal {i+1}"
        color = colors[i] if colors and i < len(colors) else None
        plt.plot(support, signal, label=label, color=color, linewidth=1.5)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

def plot_plan_spectra(support, source, target, plan, source_freq_start=0, source_freq_end=-1, target_freq_start=0, target_freq_end=-1, log=True, epsilon=1e-5):

    """
    Plot OT plan alongside spectra.

    Args:
        support (np.ndarray): Frequency support.
        source (np.ndarray): Normalized source spectrum.
        target (np.ndarray): Normalized target spectrum.
        plan (np.ndarray): OT Plan
        source_freq_start (double): Start frequency for source spectrum.
        source_freq_end (double): End frequency for source spectrum.
        target_freq_start (double): Start frequency for target spectrum.
        target_freq_end (double): End frequency for target spectrum.
        log (bool): Use log-scale to plot.
        epsilon (double): Parameter for log-scale.

    """
    
    row_start = frequency_to_index(support, source_freq_start)
    row_end = frequency_to_index(support, source_freq_end)
    col_start = frequency_to_index(support, target_freq_start)
    col_end = frequency_to_index(support, target_freq_end)

    matrix = plan[row_start:row_end, col_start:col_end]

    if log:
        matrix = np.log(matrix + epsilon)

    fig = plt.figure(figsize=(8, 8))

    gs = fig.add_gridspec(3, 3, height_ratios=[.01, 1, 6], width_ratios=[1, 6, .3])

    ax_title = fig.add_subplot(gs[0, 1])
    ax_title.axis('off')
    if log:
        ax_title.set_title('OT Plan (log scale)')
    else:
        ax_title.set_title('OT Plan')

    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', origin='upper', extent=[support[col_start], support[col_end-1], support[row_end], support[row_start]])
    ax.axis('off')

    ax_target = fig.add_subplot(gs[1, 1], sharex=ax)
    ax_target.plot(support[col_start:col_end], target[col_start:col_end], 'r')
    ax_target.set_yticks([])
    ax_target.set_xlabel('Target signal - Frequency (Hz)')

    ax_source = fig.add_subplot(gs[2, 0], sharey=ax)
    ax_source.plot(source[row_start:row_end], support[row_start:row_end], 'b')
    ax_source.set_ylabel('Source signal - Frequency (Hz)')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xticks([])

    # Set x-axis and y-axis ticks to represent frequency
    ax_target.set_xlim(support[col_start], support[col_end-1])
    ax.set_xlim(support[col_start], support[col_end-1])

    ax_source.set_ylim(support[row_end-1], support[row_start])  # This ensures that the y-axis starts at the top with the highest frequency
    ax.set_ylim(support[row_end-1], support[row_start])

    fig.colorbar(im, cax=plt.subplot(gs[2, 2]))
    
    # Adjust layout
    plt.subplots_adjust(wspace=0., hspace=0.2)
    plt.tight_layout()


    # Show the plot
    plt.show()

def audio_widget(signal, title=None, sr=44100):
    """
    Display audio widget to play audio.

    Args:
        signal (np.ndarray): Audio signal.
        title (string): Title of the signal.
        sr (int): Sample rate (in Hz).
    """

    audio_player = IPython.display.Audio(data=signal, rate=sr)
    out = widgets.Output()
    with out:
        display(audio_player)
    combined_widget = widgets.VBox([widgets.Label(title), out])

    return combined_widget