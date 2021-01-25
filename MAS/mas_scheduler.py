import torch

from MAS.adam_sgd_mix import AdamSGDWeighted


class MASScheduler:
    def __init__(self, optimizer, start=0, end=1, epochs=200):
        self.epochs = epochs
        self.start = start
        self.end = end
        self._step_count = 0
        self.optimizer = optimizer

    def step(self):
        self._step_count += 1
        if not self._step_count > self.epochs:
            if self.start > self.end:
                diff = self.start - self.end
                self.adam_w = self.start - self._step_count / self.epochs * diff
            else:
                diff = self.end - self.start
                self.adam_w = self._step_count / self.epochs * diff + self.start

            self.sgd_w = 1 - self.adam_w
            # update default
            self.optimizer.defaults['adam_w'] = self.adam_w
            self.optimizer.defaults['sgd_w'] = self.sgd_w
            # update all group of params
            for group in self.optimizer.param_groups:
                group['adam_w'] = self.adam_w
                group['sgd_w'] = self.sgd_w

        print('', self._step_count, 'adam w:', self.adam_w, 'sgd w:', self.sgd_w)


if __name__ == '__main__':
    fake_paramiters = [torch.tensor(1, dtype=torch.float32)]
    start = 0.2
    end = 0.5
    epochs = 4
    optimizer = AdamSGDWeighted(fake_paramiters, lr=0.1, adam_w=start, sgd_w=(1 - start), )
    mas_scheduler = MASScheduler(optimizer, start, end, epochs)
    for i in range(epochs + 2):
        # for group in optimizer.param_groups:
        #     print(i, 'OPT)', 'adam w:', group['adam_w'], 'sgd w:', group['sgd_w'])
        optimizer.step()
        mas_scheduler.step()
