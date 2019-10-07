import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .logger import make_logger
from scipy.stats import rankdata
from functools import partial

log = make_logger('Report')


class Report(object):
    """Report class that contains results from runnning polyssifier
    """

    def __init__(self, scores, confusions, predictions,
                 test_prob, coefficients, feature_selection,
                 scoring='auc'):
        self.scores = scores
        self.confusions = confusions
        self.predictions = predictions
        self.test_proba = test_prob
        self.coefficients = coefficients
        self.scoring = scoring
        self._feature_selection = feature_selection

    def plot_scores(self, path='temp'):
        plot_scores(self.scores, self.scoring, path)

    def plot_features(self, ntop=3, path='temp',
                      coef_names=None):
        if self._feature_selection:
            log.warning(
                'Feature importance not implemented for feature_selection=True, try setting False')
        else:
            plot_features(coefs=self.coefficients,
                          coef_names=coef_names,
                          ntop=ntop, file_name=path)


def plot_features(coefs, coef_names=None,
                  ntop=3, file_name='temp'):

    fs = {}

    for key, val in coefs.items():

        if val[0] is not None:
            val = np.array(val).squeeze()  # [folds, labels, coefs]
            if len(val.shape) == 2:
                fs[key] = val
            else:
                fs[key] = val.mean(axis=1)

    n_coefs = fs[list(fs.keys())[0]].shape[-1]
    if coef_names is None:
        coef_names = np.array([str(c + 1) for c in range(n_coefs)])
    else:
        coef_names = np.array(coef_names)

    for key, val in fs.items():

        figure_path = file_name + '_' + key + '_feature_ranking.png'
        log.info('Plotting %s coefs to %s', key, figure_path)
        plt.figure(figsize=(10, 10))
        # plotting coefficients weights
        print(key)
        mean = np.mean(val, axis=0)
        std = np.std(val, axis=0)
        idx = np.argsort(np.abs(mean))

        topm = mean[idx][-ntop:][::-1]
        tops = std[idx][-ntop:][::-1]
        plt.subplot(211)
        plt.bar(range(ntop), topm, yerr=tops,
                tick_label=(coef_names[idx][-ntop:][::-1]))
        plt.title('{}: Feature importance'.format(key))
        plt.xlabel('Feature index')

        # plotting coefficients rank
        rank = n_coefs - np.apply_along_axis(
            partial(rankdata, method='max'), axis=1, arr=np.abs(val))
        rank_mean = rank.mean(axis=0)
        rank_std = rank.std(axis=0)
        idx = np.argsort(rank_mean)
        topm = rank_mean[idx][:ntop]
        tops = rank_std[idx][:ntop]

        plt.subplot(212)
        plt.bar(range(ntop), topm, yerr=tops,
                tick_label=coef_names[idx][:ntop])
        plt.title('{}: Feature rank'.format(key))
        plt.xlabel('Feature index')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(figure_path)


def plot_scores(scores, scoring='auc', file_name='temp', min_val=None):

    df = scores.apply(np.mean).unstack().join(
        scores.apply(np.std).unstack(), lsuffix='_mean', rsuffix='_std')
    df.columns = ['Test score', 'Train score', 'Test std', 'Train std']
    df.sort_values('Test score', ascending=False, inplace=True)
    error = df[['Train std', 'Test std']]
    error.columns = ['Train score', 'Test score']
    data = df[['Train score', 'Test score']]

    nc = df.shape[0]

    ax1 = data.plot(kind='bar', yerr=error, colormap='coolwarm',
                    figsize=(nc * 2, 5), alpha=1)
    #ax1.set_axis_bgcolor((.7, .7, .7))
    # ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=2, fancybox=True, shadow=True)

    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    plt.ylabel(scoring, fontsize='large', rotation='horizontal')
    ax1.yaxis.grid(True)

    # This creates the legend for the plot
    testing_label = mpatches.Patch(color='red', label='Testing Score')
    training_label = mpatches.Patch(color='blue', label='Training Score')
    plt.legend(handles=[testing_label, training_label], loc='upper right')

    temp = np.array(data)

    # These statements check to see what scoring was used and size the y-axis of the graphical score report
    # accordingly.
    if(scoring == 'r2'):
        ymax = 1
        ymin = 0
    elif(scoring == 'mse'):
        ymin = 0
        ymax = temp.max()
    else:
        ymin = np.max(temp.min() - .1, 0) if min_val is None else min_val
        ymax = 1

    ax1.set_ylim(ymin, ymax)
    for n, rect in enumerate(ax1.patches):
        if n >= nc:
            break
        ax1.text(rect.get_x() - rect.get_width() / 2., ymin + (1 - ymin) * .01,
                 data.index[n], ha='center', va='bottom',
                 rotation='90', color='black', fontsize=15)
    plt.tight_layout()
    plt.savefig(file_name + '.pdf')
    plt.savefig(file_name + '.svg', transparent=False)
    return (ax1)
