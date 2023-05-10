
# This module is used to visualize the data and save the figures.

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick


# A function to visualize an accuracy metric of a given model on reduced and original data
def plot_data(data, xlabel, ylabel, xlabels, legends, title, target_file, display=False, save=True, percentage=True):

    # Width of the bars
    bar_width = 1 / (data.shape[0] + 1)
    # Colors of the bars
    colors = ["#064E3B", "#312E81", "#78350F", "#7F1D1D"]
    # Patterns of the bars
    patterns = ["/", "\\", "-", "|"]
    # Average of each metric
    averages = np.zeros((data.shape[0], data.shape[1]))
    # Standard Deviation of each metric
    stds = np.zeros((data.shape[0], data.shape[1]))

    # Calculate the average and standard deviation of each metric
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            averages[i, j] = data[i, j].mean()
            stds[i, j] = data[i, j].std()

    # Define the position of each bar
    positions = np.arange(len(averages[0]))

    # Init a figure object
    fig = plt.figure(figsize=(16, 9), dpi=120)

    # Plot the bars and error bars for each metric
    for i in range(data.shape[0]):
        # Plot the bars for original data metrics
        plt.bar(positions + i * bar_width, averages[i], color=colors[i], hatch=patterns[i], width=bar_width,
                label=legends[i], alpha=0.3)
        # Plot the error bars for original data metrics
        plt.errorbar(positions + i * bar_width, averages[i], yerr=stds[i], fmt="none",
                     ecolor=colors[i], capsize=10, capthick=0.5)
        # Display the values as labels for each bar
        for j in range(len(averages[i])):
            # get only 2 digits after the decimal point
            if percentage:
                label = "{:.2f}".format(averages[i, j])
                label = label + "%"
                plt.text(x=j + i * bar_width - bar_width / 2 + 0.01, y=101, s=label, fontsize=13,
                         fontweight="bold", color=colors[i])
            else:
                label = "{:.0f}".format(averages[i, j])
                yposition = averages[i, j] + stds[i, j] + (averages.max(initial=0) + stds.max(initial=0)) * 0.01
                plt.text(x=j + i * bar_width - bar_width / 2 + 0.06, y=yposition, s=label, fontsize=13,
                         fontweight="bold", color=colors[i])

    # Define labels and title
    # plt.xlabel(xlabel, fontsize=18, fontweight="bold")
    # plt.ylabel(ylabel, fontsize=18, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=22)
    if percentage:
        plt.axis([-bar_width, averages.shape[1] - bar_width, 0, 120])
    else:
        plt.axis([-bar_width, averages.shape[1] - bar_width, 0, (averages.max(initial=0) + stds.max(initial=0)) * 1.2])
    plt.xticks([r + (averages.shape[0] - 1) * 0.5 * bar_width for r in range(averages.shape[1])], xlabels)
    # plt.title(title, fontsize=16, fontweight="bold")

    # Add a legend for each bar group
    plt.legend(loc="upper left", ncol=2, fontsize="16")

    if percentage:
        # Plot a 100% thin dotted line
        plt.plot([-bar_width, averages.shape[0] + averages.shape[1] * bar_width], [100, 100], color="#000000",
                 linestyle="-.", linewidth=0.1)
        # Display data as percentages
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    # Display the figure
    if display:
        plt.show()

    # Save the figure to pdf and jpg formats
    if save:
        fig.savefig(target_file + ".pdf", format="pdf", dpi=120)
        fig.savefig(target_file + ".jpg", format="jpg", dpi=120)

    # Close the figure
    plt.close(fig)


# Set the default font family to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
