import logging
import os
import matplotlib
#matplotlib.use('agg')
from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path

class logger:
    def __init__(self, model_name, model_save_path, losses):
        self.path = os.path.join(model_save_path, "log")
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.losses = dict()
        for loss in losses:
            self.losses[str(loss)] = list()

        logging.basicConfig(filename=os.path.join(self.path, model_name + "_log.log" ), level=logging.INFO)
        self.log = logging.getLogger()


    def start(self, epochs=None, batch_size=None, learning_rate=None, val_percent=None, n_train=None, n_test=None, save_checkpoint=False, device = None, amp=False):
        self.epochs = epochs
        self.log.info(f'Model: {self.model_name}')
        self.log.info(f'Training date and time:  {str(datetime.now())}\n--------------------------------------\n')
        logging.info(f'Training parameters:\nEpochs:          {epochs}\nBatch size:      {batch_size}\nLearning rate:   {learning_rate}\nValidation percent: {val_percent}\nTraining size:   {n_train}\nValidation size: {n_test}\nCheckpoints:     {save_checkpoint}\nDevice:          {device.type}\nMixed Precision: {amp}\n--------------------------------------\n')

    def update(self, loss, epoch, val = None):
        if type(loss) == dict:
            losses = ""
            for key in loss:
                losses += f'{key} was {loss[key]:.3f} '
            if val != None:
                self.log.info(f'After epoch {epoch+1}/{self.epochs}: {losses} and validation score was {val:.3f}')
            else:
                self.log.info(f'After epoch {epoch+1}/{self.epochs}: {losses}')       
        else:
            if val != None:
                self.log.info(f'After epoch {epoch+1}/{self.epochs}: Training loss was {loss:.3f} and validation score was {val:.3f}')
            else:
                self.log.info(f'After epoch {epoch+1}/{self.epochs}: Training loss was {loss:.3f}')

    def update_loss(self, loss, loss_key, global_step):
        self.losses[loss_key].append((global_step, loss))

    def finish(self):
        print("Plotting losses...")
        iter = 0
        fig = plt.figure()
        #fig.suptitle(f'{self.model_name} {datetime.now().date()}')
        for loss in self.losses:
            iter += 1
            y = []
            x = []
            for value in self.losses[loss]:
                y.append(value[1])
                x.append(int(value[0]))
            #plt.subplots(figsize=(6,6))
            plt.subplot(len(self.losses),1, iter)
            plt.plot(x, y)
            plt.locator_params(axis="x", integer=True, tight=True)
            plt.title(loss)
        plt.tight_layout()
        fig.tight_layout()
        plt.savefig(os.path.join(self.path, self.model_name + "_losses.pdf"))

