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
from model import Linear_esm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (binary_annotations, fasta_files) in enumerate(dataloader):

        optimizer.zero_grad()

        outputs = []
        for fasta_file in fasta_files:
            output = model.forward(fasta_file).to(device)
            outputs.append(output)

        outputs = torch.stack(outputs).to(device)

        if outputs.shape != binary_annotations.shape:
            print(
                f"Error: Output shape {outputs.shape} does not match target shape {binary_annotations.shape}"
            )
            raise ValueError(
                f"Output shape {outputs.shape} does not match target shape {binary_annotations.shape}"
            )

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
    with torch.no_grad():
        for batch_idx, (binary_annotations, fasta_files) in enumerate(dataloader):
            outputs = []
            for fasta_file in fasta_files:
                output = model(fasta_file).to(device)
                outputs.append(output)
            outputs = torch.stack(outputs)

            # Compute the loss
            loss = criterion(outputs, binary_annotations)
            total_loss += loss.item()

            # Get predictions: Use threshold 0.5 for binary classification
            preds = (
                (outputs > 0.5).float()
                if outputs.size(1) == 2
                else torch.argmax(outputs, dim=1)
            )

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(binary_annotations.cpu().numpy())

    # Calculate F1 score and accuracy
    f1 = f1_score(all_targets, all_preds, average="macro")
    accuracy = accuracy_score(all_targets, all_preds)

    return total_loss / len(dataloader), f1, accuracy


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

    fasta_files = [
        f
        for f in glob.glob(os.path.join(args.fasta_dir, "*.fasta"))
        if not f.endswith("_temp.fasta")
    ]
    temp_fasta_files = glob.glob(os.path.join(args.fasta_dir, "*_temp.fasta"))
    for temp_file in temp_fasta_files:
        os.remove(temp_file)

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

    if model_location == "esm2_t48_15B_UR50D":
        num_layers = 47
    elif model_location == "esm2_t36_3B_UR50D":
        num_layers = 35
    elif model_location == "esm2_t33_650M_UR50D":
        num_layers = 32
    elif model_location == "esm2_t30_150M_UR50D":
        num_layers = 29
    elif model_location == "esm2_t12_35M_UR50D":
        num_layers = 11
    elif model_location == "esm2_t6_8M_UR50D":
        num_layers = 5
    else:
        raise ValueError(f"Unknown model location: {model_location}")

    dataset = FastaDataset(fasta_files)
    kf = KFold(n_splits=k_folds, shuffle=True)
    save_dir = args.save_dir

    fold_results = []
    train_losses = []
    val_losses = []

    early_stopping_patience = 5
    min_val_loss = np.Inf
    patience_counter = 0
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

        model = Linear_esm(
            model_location, output_dim, num_blocks, num_layers, dropout_rate
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
        for epoch in range(num_epochs):
            avg_loss = train(model, train_dataloader, criterion, optimizer, device)
            fold_train_losses.append(avg_loss)

            val_loss, f1, accuracy = validate(model, val_dataloader, criterion, device)
            fold_val_losses.append(val_loss)
            fold_f1_scores.append(f1)
            fold_accuracies.append(accuracy)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}"
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
