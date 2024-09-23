import torch
import torch.nn.functional as F
import torchmetrics
import hydra
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

from src.models.classification import HMSEEGClassifierModule, HMSSpectrClassifierModule
from src.data.datamodule import HMSSignalClassificationDataModule
from torchmetrics.regression import KLDivergence
from src.utils import get_transformations


def combine_predictions(preds_eeg, preds_spec, weight_eeg, weight_spec):
    combined_preds = (weight_eeg * preds_eeg + weight_spec * preds_spec) / 100
    return combined_preds

def evaluate_predictions(preds, targets, votes, device):
    pred_classes = preds.argmax(dim=1).cpu().numpy()   
    true_classes = targets.cpu().numpy()
    true_votes = votes.cpu().numpy() 
    
    accuracy = accuracy_score(true_classes, pred_classes)
    precision = precision_score(true_classes, pred_classes, average='weighted')
    recall = recall_score(true_classes, pred_classes, average='weighted')
    f1 = f1_score(true_classes, pred_classes, average='weighted')
    
    # preds_softmax = F.softmax(preds, dim=1).to(device)
    # votes_softmax = torch.tensor(true_votes).to(device)

    assert preds.shape == votes.shape, f"Shape mismatch: preds shape {preds.shape}, votes shape {votes.shape}"
    assert torch.all(votes > 0), "votes contiene valori negativi o zero, non validi per il calcolo della KL divergence"
    
    # print(f"Preds: {preds}")
    # print(f"Targets: {targets}")
    # print(f"Targets one hot: {targets_one_hot}")
    # print(f"Votes: {true_votes}")

    #compute the KL divergence between the predicted distribution and the true distribution
    # kl_divergence_metric = KLDivergence().to(device)
    # kl_divergence = kl_divergence_metric(preds_softmax, votes_softmax)

    preds = preds.to(device)
    targets = targets.to(device)
    votes = votes.to(device)

    log_probs = F.log_softmax(preds, dim=1)
    kl_divergence = F.kl_div(log_probs, votes, reduction='batchmean')
    
    return accuracy, precision, recall, f1, kl_divergence




@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):

    transformations = get_transformations(cfg)

    data = HMSSignalClassificationDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        transform=transformations,
        freeze=False,
        task='eegsspectr',
        highcut=cfg.dataset.highcut,
        dataset_type=cfg.dataset.dataset_type,
        augmentation=cfg.dataset.augmentation
    )

    test_loader = data.test_dataloader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eegs_model_path = f"{cfg.train.save_path}eegs_{cfg.train.eegs_run_name}.ckpt"
    spectr_model_path = f"{cfg.train.save_path}spectr_{cfg.train.spectr_run_name}.ckpt"
    model_eeg = HMSEEGClassifierModule.load_from_checkpoint(eegs_model_path).to(device)
    model_spec = HMSSpectrClassifierModule.load_from_checkpoint(spectr_model_path).to(device)

    all_labels = []
    all_votes = []
    all_preds_eeg = []
    all_preds_spec = []

    # Loop through the test data
    for batch in test_loader:
        (eeg, spectr), (labels, vote_distr) = batch
        eeg, spectr, labels, vote_distr = eeg.to(device), spectr.to(device), labels.to(device), vote_distr.to(device)

        with torch.no_grad():
            preds_model_eeg = model_eeg(eeg)
            preds_model_spec = model_spec(spectr)

        all_labels.extend(labels.cpu().numpy())
        all_votes.extend(vote_distr.cpu().numpy())
        all_preds_eeg.append(preds_model_eeg.cpu())
        all_preds_spec.append(preds_model_spec.cpu())
    

    all_labels = np.array(all_labels)
    all_votes = np.array(all_votes)
    all_preds_eeg = torch.cat(all_preds_eeg, dim=0)
    all_preds_spec = torch.cat(all_preds_spec, dim=0)

    unique, counts = np.unique(all_labels, return_counts=True)
    class_distribution = dict(zip(unique, counts / len(all_labels)))
    print("Distribuzione delle classi nel set di test:")
    for cls, proportion in class_distribution.items():
        print(f"Classe {cls}: {proportion:.2f}")

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    kl_divergences = []

    weights = [(i, 100 - i) for i in range(0, 101, 5)]

    #  for each combination of weights
    for weight_eeg, weight_spec in weights:
        combined_preds = combine_predictions(all_preds_eeg, all_preds_spec, weight_eeg, weight_spec)
        accuracy, precision, recall, f1, kl_divergence = evaluate_predictions(combined_preds, torch.tensor(all_labels).to(device), torch.tensor(all_votes).to(device), device)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        kl_divergences.append(kl_divergence.item())

        print(f"Weights {weight_eeg}-{weight_spec}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  KL Divergence: {kl_divergence:.4f}")

    tick_font_size = 14  # Dimensione font ticks
    axis_label_font_size = 16  # Dimensione font label degli assi
    title_font_size = 18  # Dimensione font titolo

    # Plot KL Divergence
    weight_combinations = [f"{w1}-{w2}" for w1, w2 in weights]
    plt.figure(figsize=(12, 6))
    plt.plot(weight_combinations, kl_divergences, marker='o')

    plt.xticks(rotation=90, fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.xlabel('Weights Combination (EEG-Spectrogram)', fontsize=axis_label_font_size)
    plt.ylabel('KL Divergence', fontsize=axis_label_font_size)
    plt.title('KL Divergence for Weights Combination', fontsize=title_font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kl_divergence_plot.png')
    plt.show()


    # Plot Accuracy, Precision, Recall, F1 Score
    plt.figure(figsize=(12, 6))

    plt.plot(weight_combinations, accuracies, marker='o', label='Accuracy')
    plt.plot(weight_combinations, precisions, marker='o', label='Precision')
    plt.plot(weight_combinations, recalls, marker='o', label='Recall')
    plt.plot(weight_combinations, f1_scores, marker='o', label='F1 Score')

    plt.xticks(rotation=90, fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.xlabel('Weights Combination (EEG-Spectrogram)', fontsize=axis_label_font_size)
    plt.ylabel('Metrics Value', fontsize=axis_label_font_size)
    plt.title('Accuracy, Precision, Recall, F1 Score for Weights Combination', fontsize=title_font_size)
    plt.legend(fontsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.show()



if __name__ == '__main__':
    main()
