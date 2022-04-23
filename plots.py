from train import *
import helpers

#Plotting
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
FIGDIR = 'fig/'

def plot_centralized_fixed_cuttoff_experiment(exp):
    f, axes = plt.subplots(1, 2, figsize=(9,4), sharey=False)
    mnist = exp[exp['benchmark'] == 'MNIST'].reset_index()
    sns.lineplot(data=mnist, x='epoch', hue='S', y='test_acc', ax=axes[0])

    cifar = exp[exp['benchmark'] == 'CIFAR'].reset_index()
    sns.lineplot(data=cifar, x='epoch', hue='S', y='test_acc', ax=axes[1])

    axes[0].set_title('MNIST', fontsize=14)
    axes[0].set_ylabel('Test accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].tick_params(labelsize=14)

    axes[1].set_title('CIFAR', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('', fontsize=14)
    axes[1].tick_params(labelsize=14)

    handles, labels = axes[1].gca().get_legend_handles_labels()
    order = [3,0,1,2]
    axes[1].legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    plt.savefig(f'{FIGDIR}exp1_centralized.png', dpi=400)

if __name__ == "__main__":
    exp, params = helpers.load_exp('central_baseline')
    exp['epoch'] = exp['epoch'] + 1
    plot_centralized_fixed_cuttoff_experiment(exp)
