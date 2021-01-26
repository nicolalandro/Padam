import os
import matplotlib.pyplot as plt

from log_analisys.folder_mean_accuracy import get_accuracies

model = 'resnet'
folder_path = f'../logs/{model}'
output_path = '../images/folder.pdf'


def main():
    plt.figure(figsize=(2, 1))
    f, (ax, axleg) = plt.subplots(2, figsize=(10, 10))

    files = os.listdir(folder_path)
    files.sort()
    for f in files:
        log_path = os.path.join(folder_path, f)
        epochs, opt_accuracies = get_accuracies(log_path, verbose=0)
        ax.plot(epochs, opt_accuracies, label=f)

    ax.set_ylabel('Test Accuracy')
    ax.set_xlabel('Epoch')
    axleg.axis('off')
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=2, mode="expand", borderaxespad=0.)
    plt.margins(0, 0)
    # plt.show()
    plt.savefig(output_path)


if __name__ == '__main__':
    main()
