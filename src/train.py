import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
import pytorch_lightning as pl

class EEGEffnetB0(pl.LightningModule):
    """Trains EEG data classification using EfficientNetB0 with KL divergence loss."""
    
    def __init__(self):
        super(EEGEffnetB0, self).__init__()
        self.base_model = efficientnet_b0(pretrained=True)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, 6)
        self.prob_out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.prob_out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        loss = kl_loss(F.log_softmax(y_hat, dim=1), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
