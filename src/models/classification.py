import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from torch import nn
import hydra


class HMSEEGClassifierModule(LightningModule):

    def __init__(self, signal_len, num_classes, lr=1e-5, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc_input_size = 512 * (
                int(signal_len) // 32)  # 512 out_channels of 5th conv layer and 32 because signal len is reducted
        # after 5 max pooling (2^5 = 32)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
        self.total_labels = None
        self.total_predictions = None
        self.classes = [i for i in range(num_classes)]

	

    def preprocess(self, x):
        # see MNE lib

        # drop EKG

        # normalize label votes
        # labels = df[vote_cols].values
        # labels = torch.from_numpy(labels).double()
        # labels = labels / labels.sum(dim=1, keepdim=True) # Normalize vote ratios

        # see Hz characteristics

        return x

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)

        x = x.view(-1, self.fc_input_size)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.softmax(x)

        return x

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        eegs, labels = batch
        x = self.preprocess(eegs)
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)
        self.log('test_accuracy', accuracy_score(labels, predictions), on_step=False, on_epoch=True, logger=True)
        self.log('recall', recall_score(labels, predictions, average='micro'), on_step=False, on_epoch=True, logger=True)
        self.log('precision', precision_score(labels, predictions, average='micro'), on_step=False, on_epoch=True,
                 logger=True)
        self.log('f1', f1_score(labels, predictions, average='micro'), on_step=False, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        eeg, label = batch
        x = self.preprocess(eeg)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _common_step(self, batch, batch_idx, stage):
        signals, labels = batch
        signals = self.preprocess(signals)

        pred = self(signals)
        loss = self.loss(pred, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)

        return loss



class HMSSpectrClassifierModule(LightningModule):

    def __init__(self, img_size, num_classes, lr=1e-5, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc_input_size = 512 * (img_size // 32) ** 2  # 512 out_channels of 5th conv layer and 32 because signal len is reducted
        # after 5 max pooling (2^5 = 32)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def preprocess(self, x):
        # Qui puoi implementare la tua logica di preelaborazione per le immagini PNG
        return x

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)

        x = x.view(-1, self.fc_input_size)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.softmax(x)

        return x

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        images, labels = batch
        x = self.preprocess(images)
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)
        self.log('test_accuracy', accuracy_score(labels, predictions), on_step=False, on_epoch=True, logger=True)
        self.log('recall', recall_score(labels, predictions, average='micro'), on_step=False, on_epoch=True, logger=True)
        self.log('precision', precision_score(labels, predictions, average='micro'), on_step=False, on_epoch=True,
                 logger=True)
        self.log('f1', f1_score(labels, predictions, average='micro'), on_step=False, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, label = batch
        x = self.preprocess(images)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _common_step(self, batch, batch_idx, stage):
        images, labels = batch
        images = self.preprocess(images)

        pred = self(images)
        loss = self.loss(pred, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)

        return loss
