import torch
import torch.nn.functional as F

class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.mlp = torch.nn.Sequential(torch.nn.Linear(5*5, 32), torch.nn.ReLU(),
                                          torch.nn.Linear(32, 512), torch.nn.ReLU(),
                                          torch.nn.Linear(512, 5*2))
    

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        t = torch.tensor(x.reshape(x.shape[0], -1))
        return self.mlp(t)


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'mlp.th'))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
