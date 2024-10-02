import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import glob
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from protein_embedding import ProteinEmbedding
from dataset import FastaDataset
from model import ECT
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (binary_annotations, fasta_files) in enumerate(dataloader):

        optimizer.zero_grad()

        binary_annotations = binary_annotations.to(device)

        outputs = model(fasta_files)

        loss = criterion(outputs, binary_annotations)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []  # To store probabilities or scores for AUC
    with torch.no_grad():
        for batch_idx, (binary_annotations, fasta_files) in enumerate(dataloader):
            binary_annotations = binary_annotations.to(device)
            outputs = model(fasta_files)

            # Compute the loss
            loss = criterion(outputs, binary_annotations)
            total_loss += loss.item()

            # Convert one-hot encoded targets to class indices if necessary
            if binary_annotations.ndim > 1 and binary_annotations.size(1) > 1:
                binary_annotations = torch.argmax(binary_annotations, dim=1)

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(binary_annotations.cpu().numpy())

            # Store probabilities for AUC calculation
            if outputs.size(1) == 2:
                # Binary classification: store probability of positive class
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                # Multi-class classification: append probability distributions
                all_probs.append(probs.cpu().numpy())

    # Convert lists to numpy arrays
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Process all_probs for AUC calculation
    if outputs.size(1) == 2:
        all_probs = np.array(all_probs)  # Shape: (num_samples,)
    else:
        all_probs = np.concatenate(all_probs, axis=0)  # Shape: (num_samples, num_classes)

    # Calculate F1 score and accuracy
    f1 = f1_score(all_targets, all_preds, average="macro")
    accuracy = accuracy_score(all_targets, all_preds)

    # Calculate AUC
    if outputs.size(1) == 2:
        # Binary classification
        auc = roc_auc_score(all_targets, all_probs)
    else:
        # Multi-class classification
        auc = roc_auc_score(all_targets, all_probs, average="macro", multi_class="ovr")

    return total_loss / len(dataloader), f1, accuracy, auc

def custom_collate_fn(batch, output_dim):
    annotations, fasta_files = zip(*batch)

    batch_size = len(annotations)

    binary_annotations = torch.zeros(batch_size, output_dim, dtype=torch.float).to(
        device
    )

    for i, ann_list in enumerate(annotations):
        for ann in ann_list:
            binary_annotations[i, ann] = 1.0

    return binary_annotations, fasta_files


def main():
    parser = argparse.ArgumentParser(description="Train your model with specific data")
    parser.add_argument(
        "model_location", type=str, help="The name of the ESM BERT model"
    )
    parser.add_argument("fasta_dir", type=str, help="Directory containing FASTA files")
    parser.add_argument(
        "save_dir", type=str, help="Directory for saving model checkpoints"
    )
    parser.add_argument("output_dim", type=int, help="Number of groups to classify")
    parser.add_argument("num_blocks", type=int, help="Number of linear blocks to use")
    parser.add_argument("batch_size", type=int, help="Batch size to use")
    parser.add_argument("learning_rate", type=float, help="Learning rate to use")
    parser.add_argument("num_epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "--dropout_rate", type=float, default=0, help="Dropout rate to use"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="weight_decay value"
    )
    parser.add_argument("--nogpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()

    # Now use args.model_location, args.fasta_dir, etc., in your script logic
    print(f"Model location: {args.model_location}")
    print(f"FASTA directory: {args.fasta_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Output dimension: {args.output_dim}")
    print(f"Number of blocks: {args.num_blocks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Use GPU: {not args.nogpu}")
    print(f"Dropout_rate: {args.dropout_rate}")
    print(f"Weight Decay: {args.weight_decay}")


    print("Handling the fasta files in custom dataset")
    fasta_files = [
        f
        for f in glob.glob(os.path.join(args.fasta_dir, "*.fasta"))
        if not f.endswith("_temp.fasta")
    ]
    temp_fasta_files = glob.glob(os.path.join(args.fasta_dir, "*_temp.fasta"))
    for temp_file in temp_fasta_files:
        os.remove(temp_file)
    print("Handling fasta files end")
    ########## Training Parameters #################
    output_dim = args.output_dim
    num_blocks = args.num_blocks
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    dropout_rate = args.dropout_rate
    weight_decay = args.weight_decay
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.nogpu else "cpu"
    )
    k_folds = 5
    #################################################

    ################ 5-Cross Validation #############
    model_location = args.model_location


    print("Handling FastaDataset")
    dataset = FastaDataset(fasta_files)
    print("Handling FastaDataset End")
    kf = KFold(n_splits=k_folds, shuffle=True)
    save_dir = args.save_dir

    fold_results = []
    train_losses = []
    val_losses = []

    early_stopping_patience = 5
    min_val_loss = np.Inf
    patience_counter = 0
    print("Training starts")
    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        if fold != 0:  # Skip all folds except the first one
            continue

        print(f"FOLD {fold+1}/{k_folds}")
        print("--------------------------------")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_dataloader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: custom_collate_fn(batch, output_dim),
        )
        val_dataloader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: custom_collate_fn(batch, output_dim),
        )

        model = ECT(
            model_location, output_dim, num_blocks, dropout_rate
        ).to(device)

        if output_dim == 2:
            criterion = nn.BCELoss().to(device)
        elif output_dim > 2:
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            raise ValueError(
                "Invalid output_dim. It should be 2 for binary classification or greater than 2 for multi-class classification."
            )
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        fold_train_losses = []
        fold_val_losses = []
        fold_f1_scores = []
        fold_accuracies = []
        print("Training epoch starts")
        for epoch in range(num_epochs):
            avg_loss = train(model, train_dataloader, criterion, optimizer, device)
            fold_train_losses.append(avg_loss)
            val_loss, f1, accuracy, auc = validate(model, val_dataloader, criterion, device)
            fold_val_losses.append(val_loss)
            fold_f1_scores.append(f1)
            fold_accuracies.append(accuracy)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}"
            )

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                print(
                    f"Early stopping counter: {patience_counter} / {early_stopping_patience}"
                )
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered due to lack of improvement.")
                    break

        if "best_model_state" in locals():
            model_save_path = os.path.join(
                save_dir, f"{model_location}_{num_blocks}_{learning_rate}_{dropout_rate}_{weight_decay}_earlystopped.pth"
            )
            torch.save(best_model_state, model_save_path)
        else:
            model_save_path = os.path.join(
                save_dir, f"{model_location}_{num_blocks}_{learning_rate}_{dropout_rate}_{weight_decay}.pth"
            )
            torch.save(model.state_dict(), model_save_path)

        train_losses.append(fold_train_losses)
        val_losses.append(fold_val_losses)

        print(
            f"Cross-validation complete for fold {fold+1}. Validation Loss = {val_loss:.4f}, F1 Score = {f1:.4f}, Accuracy = {accuracy:.4f}"
        )

        # Plotting for just this fold
        plt.plot(train_losses[0], label=f"Fold {fold+1} Training Loss")
        plt.plot(val_losses[0], label=f"Fold {fold+1} Validation Loss")
        plt.plot(fold_f1_scores, label=f"Fold {fold+1} F1 Score")
        plt.plot(fold_accuracies, label=f"Fold {fold+1} Accuracy")

        plt.title(
            f"Training and Validation Metrics for {model_location}_{num_blocks}_{learning_rate}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.legend()

        plot_save_path = os.path.join(
            save_dir,
            f"{model_location}_{num_blocks}_{learning_rate}_{dropout_rate}_{weight_decay}.png",
        )
        plt.savefig(plot_save_path)

        plt.show()
        print(f"Loss and metrics plot saved to {plot_save_path}")


if __name__ == "__main__":
    main()
