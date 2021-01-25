import os

model = 'resnet'
folder_path = f'../logs/{model}'


def get_accuracies(log_path, verbose=True):
    with open(log_path, "r") as f:
        logs = str(f.readlines())
    tests = logs.split('Epoch: ')[1:]
    opt_accuracies = []
    epochs = []
    for t in tests:
        epoch = t.split(' ')[0]
        epochs.append(int(epoch))
        if verbose:
            print('Epoch:', epoch)

        acc = t.split('Acc: ')[-1].split('%')[0].strip()
        if verbose:
            print(' ', 'accuracy:', acc)
        opt_accuracies.append(float(acc))
    return epochs, opt_accuracies


def main():
    files = os.listdir(folder_path)

    all_best_accuracies = []

    files.sort()
    old_f = 'none'
    for f in files:
        clean_f = f[:-6] if f[:-4].endswith('_1') else f[:-4]
        if not clean_f.startswith(old_f):
            old_f = clean_f
            all_best_accuracies.append([])
            index = len(all_best_accuracies) - 1
        log_path = os.path.join(folder_path, f)

        epochs, opt_accuracies = get_accuracies(log_path, verbose=0)
        best_acc = max(opt_accuracies)
        all_best_accuracies[index].append((f, best_acc))

    all_best_accuracies_mean = []
    for r in all_best_accuracies:
        runs = len(r)
        if runs > 0:
            mean_acc = sum([x[1] for x in r]) / runs
            name = r[0][0][:-4]
            all_best_accuracies_mean.append((name, mean_acc, runs))

    all_best_accuracies_mean.sort(key=lambda tup: tup[1], reverse=True)
    for i, (n, a, r) in enumerate(all_best_accuracies_mean):
        print(f'{i})', n, f'({r} runs)', '-->', a)


if __name__ == '__main__':
    main()
