import torch
import os
from pathlib import Path

class Checkpoint_saver:

    def __init__(self, epoch, model, optimizer, checkpoint_dir,tb_dir, losses=None):
        self.step = 0
        self.epoch = epoch +1
        self.path = checkpoint_dir
        self.losses = losses
        if type(model) ==tuple:
            self.model = []
            for net in model:
                self.model.append(net.state_dict())
        else:
            self.model = model.state_dict()

        if type(optimizer) ==tuple:
            self.optimizer = []
            for op in optimizer:
                self.optimizer.append(op.state_dict())
        else:
            self.optimizer = optimizer.state_dict()
        self.tb_dir = tb_dir
        
    def update(self, epoch, model, optimizer, step, losses=None):
        self.step = step
        self.epoch = epoch +1
        self.losses = losses
        if type(model) ==tuple:
            self.model = []
            for net in model:
                self.model.append(net.state_dict())
        else:
            self.model = model.state_dict()

        if type(optimizer) ==tuple:
            self.optimizer = []
            for op in optimizer:
                self.optimizer.append(op.state_dict())
        else:
            self.optimizer = optimizer.state_dict()
        
    def save_training(self):
        print("Saving the checkpoint...")
        state = {
                'epoch': self.epoch,
                'state_dict': self.model,
                'optimizer': self.optimizer,
                'tensorboard': self.tb_dir,
                'model_save_dir': self.path,
                'losses': self.losses,
                'step': self.step
                }
        save_location = os.path.join(self.path, 'PAUSE')
        Path(save_location).mkdir(parents=True, exist_ok=True)
        torch.save(state, os.path.join(save_location, "paused_training.pth"))

def load_paused_training(checkpoint_dir, model, optimizer):
    checkpoint = torch.load(checkpoint_dir)
    if type(model) ==list:
        i = 0
        for net in checkpoint['state_dict']:
            model[i].load_state_dict(net)
            model[i].train()
            i += 1
    else:  
        model.load_state_dict(checkpoint['state_dict'])
        model.train()
    if type(optimizer) ==list:
        i = 0
        for op in checkpoint['optimizer']:
            optimizer[i].load_state_dict(op)
            i+=1
    else:  
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['tensorboard'], checkpoint['model_save_dir'], checkpoint['losses'], checkpoint['step']
