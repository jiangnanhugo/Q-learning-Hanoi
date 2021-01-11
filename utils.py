import matplotlib.pyplot as plt


def plot_results(episodes, means_averaged, stds_averaged, num_of_disks, fname="result.pdf"):
    fig = plt.figure()
    plt.loglog(episodes, means_averaged, 'b.-', label='Averaged performance')
    plt.loglog(episodes, means_averaged + stds_averaged, 'b', alpha=0.5)
    plt.loglog(episodes, means_averaged - stds_averaged, 'b', alpha=0.5)
    plt.fill_between(episodes, means_averaged - stds_averaged, means_averaged + stds_averaged, facecolor='blue',
                     alpha=0.5)
    optimum_moves = 2 ** num_of_disks - 1
    plt.axhline(y=optimum_moves, color='g', label='Optimum (=%s moves)' % optimum_moves)
    plt.xlabel('Number of training episodes')
    plt.ylabel('Number of moves')
    plt.grid('on', which='both')
    plt.title('Q-learning the Towers of Hanoi game with {} discs'.format(num_of_disks))
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels)
    plt.savefig(fname)
