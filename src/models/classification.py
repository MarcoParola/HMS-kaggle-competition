import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torchmetrics.classification
import torch.nn.functional as F
from torchmetrics.regression import KLDivergence


class HMSEEGClassifierModule(LightningModule):

    def __init__(self, signal_len, num_classes, lr=1e-5, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # self.conv1 = nn.Conv1d(in_channels=11, out_channels=32, kernel_size=3, stride=1, padding=1)
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

        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.recall = torchmetrics.classification.Recall(task="multiclass", average='weighted', num_classes=num_classes)
        self.precision = torchmetrics.classification.Precision(task="multiclass", average='weighted', num_classes=num_classes)
        self.f1 = torchmetrics.classification.F1Score(task="multiclass", average='weighted', num_classes=num_classes)
        self.kl_divergence = KLDivergence()

        with open('submission.csv', 'w') as f:
            f.write('seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\n')


    def preprocess(self, x):

        return x

    def extract_features(self, x):
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
        return x

    def forward(self, x):
        # print("Features shape input", x.shape) #[32, 20, 2000]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)        # after pool -> [32, 512, 62]

        x = x.view(-1, self.fc_input_size)  # before fc1 -> # [32, 31744]

        x = self.relu(self.fc1(x))  # Features shape torch.Size([32, 128])     4.6 M Trainable params
        x = self.fc2(x)

        x = self.softmax(x)

        return x

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        eegs, (labels, vote_distr) = batch
        
        x = self.preprocess(eegs)
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)

        y_hat_np = y_hat.detach().cpu().numpy()

        with open('submission.csv', 'a') as f:
            for probs in y_hat_np:
                probs_str = ','.join(map(str, probs))
                f.write(f'{probs_str}\n')

        assert y_hat.shape == vote_distr.shape, f"Shape mismatch: y_hat shape {y_hat.shape} and vote_distr shape {vote_distr.shape} must be the same"

        #log metrics
        self.log('test_accuracy', self.accuracy(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_recall', self.recall(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_precision', self.precision(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_f1', self.f1(predictions, labels), on_step=False, on_epoch=True, logger=True)

        assert y_hat.shape == vote_distr.shape, f"Shape mismatch: y_hat shape {y_hat.shape}, vote_distr shape {vote_distr.shape}"

        log_probs = F.log_softmax(y_hat, dim=1)  
        assert torch.all(vote_distr > 0), "vote_distr contiene valori negativi o zero, non validi per il calcolo della KL divergence"
        kl_div = F.kl_div(log_probs, vote_distr, reduction='batchmean')  
        self.log('test_kl_div_log', kl_div, on_step=False, on_epoch=True, logger=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        eegs, (labels, vote_distr) = batch
        x = self.preprocess(eegs)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def _common_step(self, batch, batch_idx, stage):
        eegs, (labels, vote_distr) = batch
        eegs= self.preprocess(eegs)

        pred = self(eegs)
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
        self.fc_input_size = 512 * (
                img_size // 32) ** 2  # 512 out_channels of 5th conv layer and 32 because signal len is reducted
        # after 5 max pooling (2^5 = 32)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
        self.classes = [i for i in range(num_classes)]


        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.recall = torchmetrics.classification.Recall(task="multiclass", average='weighted', num_classes=num_classes)
        self.precision = torchmetrics.classification.Precision(task="multiclass", average='weighted', num_classes=num_classes)
        self.f1 = torchmetrics.classification.F1Score(task="multiclass", average='weighted', num_classes=num_classes)
        self.kl_divergence = KLDivergence()


    def preprocess(self, x):
        return x

    def extract_features(self, x):
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
        #print("Features shape", x.shape) # Features shape torch.Size([32, 128])     18.3 M Trainable params
        x = self.fc2(x)

        x = self.softmax(x)

        return x

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        images, (labels, vote_distr) = batch
        x = self.preprocess(images)
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)

        y_hat_np = y_hat.detach().cpu().numpy()

        with open('submission.csv', 'a') as f:
            for probs in y_hat_np:
                probs_str = ','.join(map(str, probs))
                f.write(f'{probs_str}\n')

        #log metrics
        self.log('test_accuracy', self.accuracy(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_recall', self.recall(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_precision', self.precision(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_f1', self.f1(predictions, labels), on_step=False, on_epoch=True, logger=True)

        assert y_hat.shape == vote_distr.shape, f"Shape mismatch: y_hat shape {y_hat.shape}, vote_distr shape {vote_distr.shape}"

        log_probs = F.log_softmax(y_hat, dim=1)  
        assert torch.all(vote_distr > 0), "vote_distr contiene valori negativi o zero, non validi per il calcolo della KL divergence"
        kl_div = F.kl_div(log_probs, vote_distr, reduction='batchmean')  
        self.log('test_kl_div_log', kl_div, on_step=False, on_epoch=True, logger=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, label = batch
        x = self.preprocess(images)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def _common_step(self, batch, batch_idx, stage):
        images, (labels, vote_distr) = batch
        images = self.preprocess(images)

        pred = self(images)
        loss = self.loss(pred, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)

        return loss


class HMSEEGSpectrClassifierModule(LightningModule):

    def __init__(self, num_classes, eegs_model_path = "", spectr_model_path = "", freeze=True, lr=1e-5, max_epochs=100, feat_comb_mode=None):
        super().__init__()
        self.save_hyperparameters()

        # load eeg model
        self.eeg_model = HMSEEGClassifierModule.load_from_checkpoint(eegs_model_path)
        # load spectr model
        self.spectr_model = HMSSpectrClassifierModule.load_from_checkpoint(spectr_model_path)

        self.freeze = freeze
        self.feature_comb_mode = feat_comb_mode
        
        if feat_comb_mode == 'concat':
            self.fc1 = nn.Linear(256, 128)
        else:
            self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
        self.classes = [i for i in range(num_classes)]


        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.recall = torchmetrics.classification.Recall(task="multiclass", average='weighted', num_classes=num_classes)
        self.precision = torchmetrics.classification.Precision(task="multiclass", average='weighted', num_classes=num_classes)
        self.f1 = torchmetrics.classification.F1Score(task="multiclass", average='weighted', num_classes=num_classes)
        self.kl_divergence = KLDivergence()


    def preprocess(self, x):
        return x

    def forward(self, x):
        eeg_features, spectr_features = x

        if self.freeze:

            # print(f"EEG features shape: {eeg_features.shape}")		#(32, 128)
            # print(f"Spectrogram features shape: {spectr_features.shape}")    #(32,128)
        

            # switch self.feature_comb_mode
            if self.feature_comb_mode == 'concat':
                combined_features = torch.cat((eeg_features, spectr_features), dim=1)
            elif self.feature_comb_mode == 'sum':
                combined_features = eeg_features + spectr_features
            elif self.feature_comb_mode == 'subtract':
                combined_features = eeg_features - spectr_features
            elif self.feature_comb_mode == 'mul':
                combined_features = eeg_features * spectr_features
            elif self.feature_comb_mode == 'w_sum_eeg':
                combined_features = 0.7 * eeg_features + 0.3 * spectr_features
            elif self.feature_comb_mode == 'w_sum_spectr':
                combined_features = 0.3 * eeg_features + 0.7 * spectr_features
            else:
                raise ValueError("Invalid feature combination mode")

            combined_features = combined_features.float()

            # print(f"Combined features shape: {combined_features.shape}")    #(32,256)
            # # print combined features type
            # print(f"Combined features type: {type(combined_features)}")    #<class 'torch.Tensor'>
            # # print combined features data type
            # print(f"Combined features data type: {combined_features.dtype}") 

            # print("EEG features: ", eeg_features)
            # print("Spectrogram features: ", spectr_features)
            # print("Combined features: ", combined_features)


        else:   
            eeg_features = self.eeg_model.extract_features(eeg_features)
            spectr_features = self.spectr_model.extract_features(spectr_features)

            # print(f"EEG features shape: {eeg_features.shape}")		#(32, 128)
            # print(f"Spectrogram features shape: {spectr_features.shape}")    #(32,128)


            # switch self.feature_comb_mode
            if self.feature_comb_mode == 'concat':
                combined_features = torch.cat((eeg_features, spectr_features), dim=1)
            elif self.feature_comb_mode == 'sum':
                combined_features = eeg_features + spectr_features
            elif self.feature_comb_mode == 'subtract':
                combined_features = eeg_features - spectr_features
            elif self.feature_comb_mode == 'mul':
                combined_features = eeg_features * spectr_features
            elif self.feature_comb_mode == 'weighted_sum':
                combined_features = 0.7 * eeg_features + 0.3 * spectr_features
            else:
                raise ValueError("Invalid feature combination mode")

            combined_features = combined_features.float()

            # print(f"Combined features shape: {combined_features.shape}")    (32,256)
            # # print combined features type
            # print(f"Combined features type: {type(combined_features)}")    #<class 'torch.Tensor'>
            # # print combined features data type
            # print(f"Combined features data type: {combined_features.dtype}")    #torch.float32

        out = self.fc1(combined_features)
        out = self.fc2(out)
        out = self.softmax(out)
    
        return out

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        (eeg, spectr), (labels, vote_distr) = batch
        data = (eeg, spectr)
        x = self.preprocess(data)
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)

        #log metrics
        self.log('test_accuracy', self.accuracy(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_recall', self.recall(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_precision', self.precision(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_f1', self.f1(predictions, labels), on_step=False, on_epoch=True, logger=True)
        
        assert y_hat.shape == vote_distr.shape, f"Shape mismatch: y_hat shape {y_hat.shape}, vote_distr shape {vote_distr.shape}"

        log_probs = F.log_softmax(y_hat, dim=1)
        assert torch.all(vote_distr > 0), "vote_distr contiene valori negativi o zero, non validi per il calcolo della KL divergence"
        kl_div = F.kl_div(log_probs, vote_distr, reduction='batchmean') 
        self.log('test_kl_div_log', kl_div, on_step=False, on_epoch=True, logger=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        (eeg, spectr), (labels, vote_distr) = batch
        data = (eeg, spectr)
        x = self.preprocess(data)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,
                                                               eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def _common_step(self, batch, batch_idx, stage):
        (eeg, spectr), (labels, vote_distr) = batch
        data = (eeg, spectr)
        data = self.preprocess(data)

        pred = self(data)
        loss = self.loss(pred, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)

        return loss
    
class HMSEEGSpectrClassifierModuleWithProbabilities(LightningModule):

    def __init__(self, num_classes, eegs_model_path = "", spectr_model_path = "", freeze=True, lr=1e-5, max_epochs=100, feat_comb_mode=None):
        super().__init__()
        self.save_hyperparameters()

        # load eeg model
        self.eeg_model = HMSEEGClassifierModule.load_from_checkpoint(eegs_model_path)
        # load spectr model
        self.spectr_model = HMSSpectrClassifierModule.load_from_checkpoint(spectr_model_path)

        self.freeze = freeze
        self.feature_comb_mode = feat_comb_mode
        
        if feat_comb_mode == 'concat':
            self.fc1 = nn.Linear(256, 128)
        else:
            self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.softmax = nn.Softmax(dim=1)
        # self.loss = nn.CrossEntropyLoss()
        self.classes = [i for i in range(num_classes)]


        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.recall = torchmetrics.classification.Recall(task="multiclass", average='weighted', num_classes=num_classes)
        self.precision = torchmetrics.classification.Precision(task="multiclass", average='weighted', num_classes=num_classes)
        self.f1 = torchmetrics.classification.F1Score(task="multiclass", average='weighted', num_classes=num_classes)
        self.kl_divergence = KLDivergence()


    def preprocess(self, x):
        return x

    def forward(self, x):
        eeg_features, spectr_features = x

        if self.freeze:

            # print(f"EEG features shape: {eeg_features.shape}")		#(32, 128)
            # print(f"Spectrogram features shape: {spectr_features.shape}")    #(32,128)
        

            # switch self.feature_comb_mode
            if self.feature_comb_mode == 'concat':
                combined_features = torch.cat((eeg_features, spectr_features), dim=1)
            elif self.feature_comb_mode == 'sum':
                combined_features = eeg_features + spectr_features
            elif self.feature_comb_mode == 'subtract':
                combined_features = eeg_features - spectr_features
            elif self.feature_comb_mode == 'mul':
                combined_features = eeg_features * spectr_features
            elif self.feature_comb_mode == 'w_sum_eeg':
                combined_features = 0.7 * eeg_features + 0.3 * spectr_features
            elif self.feature_comb_mode == 'w_sum_spectr':
                combined_features = 0.3 * eeg_features + 0.7 * spectr_features
            else:
                raise ValueError("Invalid feature combination mode")

            combined_features = combined_features.float()

            # print(f"Combined features shape: {combined_features.shape}")    #(32,256)
            # # print combined features type
            # print(f"Combined features type: {type(combined_features)}")    #<class 'torch.Tensor'>
            # # print combined features data type
            # print(f"Combined features data type: {combined_features.dtype}") 

            # print("EEG features: ", eeg_features)
            # print("Spectrogram features: ", spectr_features)
            # print("Combined features: ", combined_features)


        else:   
            eeg_features = self.eeg_model.extract_features(eeg_features)
            spectr_features = self.spectr_model.extract_features(spectr_features)

            # print(f"EEG features shape: {eeg_features.shape}")		#(32, 128)
            # print(f"Spectrogram features shape: {spectr_features.shape}")    #(32,128)


            # switch self.feature_comb_mode
            if self.feature_comb_mode == 'concat':
                combined_features = torch.cat((eeg_features, spectr_features), dim=1)
            elif self.feature_comb_mode == 'sum':
                combined_features = eeg_features + spectr_features
            elif self.feature_comb_mode == 'subtract':
                combined_features = eeg_features - spectr_features
            elif self.feature_comb_mode == 'mul':
                combined_features = eeg_features * spectr_features
            elif self.feature_comb_mode == 'weighted_sum':
                combined_features = 0.7 * eeg_features + 0.3 * spectr_features
            else:
                raise ValueError("Invalid feature combination mode")

            combined_features = combined_features.float()

            # print(f"Combined features shape: {combined_features.shape}")    (32,256)
            # # print combined features type
            # print(f"Combined features type: {type(combined_features)}")    #<class 'torch.Tensor'>
            # # print combined features data type
            # print(f"Combined features data type: {combined_features.dtype}")    #torch.float32

        out = self.fc1(combined_features)
        out = self.fc2(out)
        out = self.softmax(out)
    
        return out

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        (eeg, spectr), (labels, vote_distr) = batch
        data = (eeg, spectr)
        x = self.preprocess(data)
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)

        #log metrics
        self.log('test_accuracy', self.accuracy(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_recall', self.recall(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_precision', self.precision(predictions, labels), on_step=False, on_epoch=True, logger=True)
        self.log('test_f1', self.f1(predictions, labels), on_step=False, on_epoch=True, logger=True)
        
        assert y_hat.shape == vote_distr.shape, f"Shape mismatch: y_hat shape {y_hat.shape}, vote_distr shape {vote_distr.shape}"

        log_probs = F.log_softmax(y_hat, dim=1)
        assert torch.all(vote_distr > 0), "vote_distr contiene valori negativi o zero, non validi per il calcolo della KL divergence"
        kl_div = F.kl_div(log_probs, vote_distr, reduction='batchmean') 
        self.log('test_kl_div_log', kl_div, on_step=False, on_epoch=True, logger=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        (eeg, spectr), (labels, vote_distr) = batch
        data = (eeg, spectr)
        x = self.preprocess(data)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,
                                                               eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def _common_step(self, batch, batch_idx, stage):
        (eeg, spectr), (labels, vote_distr) = batch
        data = (eeg, spectr)
        data = self.preprocess(data)

        pred = self(data)
        log_probs = F.log_softmax(pred, dim=1)
        assert torch.all(vote_distr > 0), "vote_distr contiene valori negativi o zero, non validi per il calcolo della KL divergence"
        loss = F.kl_div(log_probs, vote_distr, reduction='batchmean') 
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)

        return loss
    
