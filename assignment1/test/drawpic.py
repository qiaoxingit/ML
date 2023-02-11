from matplotlib import pyplot as plt

alg = ['DT', 'KNN', 'NN', 'SVM', 'Boost']
f1_scores = [{'title': 'Breast Cancer', 'data': [0.95, 0.94, 0.95, 0.95, 0.95]},
             {'title': 'Wine Quality', 'data': [0.81, 0.80, 0.62, 0.71, 0.78]}]
f1_score_info = {'fig_location': 'assignment1/images/f1_score',
                 'title': 'Comparison of All Models Across Both Datasets',
                 'xlable': 'F1 Score'}


def plot_f1_scores():
    # Create two subplots
    fig, axses = plt.subplots(1, 2, figsize=(9, 5), sharey=True)

    # Set figure title and padding
    fig.suptitle(f1_score_info['title'])
    fig.text(0.5, 0.02, f1_score_info['xlable'])
    fig.subplots_adjust(top=0.85, bottom=0.15)

    # Plot them
    for i in range(0, 2):
        axs = axses[i]
        axs.set_xlim(0.5, 1.0)  # set x axis scale
        axs.scatter(f1_scores[i]['data'], alg, marker='o')  # scatter chart
        axs.set_title(f1_scores[i]['title'])  # set subplot's title
        axs.grid()  # show grid

    plt.savefig(f1_score_info['fig_location'])
    plt.clf()


if __name__ == "__main__":
    plot_f1_scores()
