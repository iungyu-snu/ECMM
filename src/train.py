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
    with torch.no_grad():
        for batch_idx, (binary_annotations, fasta_files) in enumerate(dataloader):

            # binary_annotations is already in the correct shape, so no need to modify it

            outputs = []
            for fasta_file in fasta_files:
                output = model(fasta_file).to(device)
                outputs.append(output)

            # Stack outputs to form a tensor with shape (batch_size, num_labels)
            outputs = torch.stack(outputs)

            # Compute the loss
            loss = criterion(outputs, binary_annotations)
            total_loss += loss.item()

    return total_loss / len(dataloader)


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

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
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

        # Initialize the model
        model = Linear_esm(model_location, output_dim, num_blocks, num_layers).to(device)

        # Define loss function and optimizer

        if output_dim == 2:
            criterion = nn.BCELoss().to(device)
        elif output_dim > 2:
            criterion = nn.CrossEntropyLoss().to(device) 
        else:
            raise ValueError("Invalid output_dim. It should be 2 for binary classification or greater than 2 for multi-class classification.")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        fold_train_losses = []
        fold_val_losses = []
        for epoch in range(num_epochs):
            avg_loss = train(model, train_dataloader, criterion, optimizer, device)
            fold_train_losses.append(avg_loss)

            val_loss = validate(model, val_dataloader, criterion, device)
            fold_val_losses.append(val_loss)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

        train_losses.append(fold_train_losses)
        val_losses.append(fold_val_losses)

        # Save the model for this fold
        model_save_path = os.path.join(save_dir, f"linear_esm_model_fold_{fold+1}.pth")
        torch.save(model.state_dict(), model_save_path)

        fold_results.append(val_loss)

    # Print cross-validation results
    print("Cross-validation complete. Results:")
    for fold, result in enumerate(fold_results):
        print(f"Fold {fold+1}: Validation Loss = {result:.4f}")
    print(f"Average Validation Loss: {sum(fold_results)/len(fold_results):.4f}")

    # Plotting the losses
    for fold in range(k_folds):
        plt.plot(train_losses[fold], label=f"Fold {fold+1} Training Loss")
        plt.plot(val_losses[fold], label=f"Fold {fold+1} Validation Loss")

    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Save the plot
    plot_save_path = os.path.join(save_dir, "training_validation_losses.png")
    plt.savefig(plot_save_path)

    # Show the plot
    plt.show()

    print(f"Loss plot saved to {plot_save_path}")


if __name__ == "__main__":
    main()
