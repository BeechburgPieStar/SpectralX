import os
import argparse
import torch
import numpy as np
import random
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils.get_dataset import TrainDataset, TestDataset, setup_seed
from utils.SpectralX import SpectralX
from torchsummary import summary


def get_param_value(model_size: str) -> int:
    """Returns the parameter value based on the input size: S, M, or L."""
    model_size_mapping = {'S': 4, 'M': 8, 'L': 16}
    if model_size in model_size_mapping:
        return model_size_mapping[model_size]
    else:
        raise ValueError(f"Invalid model_size: {model_size}. Use 'S', 'M', or 'L'.")


def get_save_path(model_size: str, use_asb: bool) -> str:
    """
    Generate save path based on model size and ASB usage.
    
    Examples:
        - SpectralX_S with ASB -> weights/SpectralX_S.pth
        - SpectralX_S without ASB -> weights/SpectralX_S_wo_ASB.pth
    """
    if use_asb:
        return f"weights/SpectralX_{model_size}.pth"
    else:
        return f"weights/SpectralX_{model_size}_wo_ASB.pth"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SpectralX Multi-Task Learning: Activity Recognition + Indoor Localization")
    
    # Mode and model
    parser.add_argument("--mode", type=str, default="train_test", choices=["train", "test", "train_test"],
                        help="Choose mode: 'train', 'test', or 'train_test'.")
    parser.add_argument("--model_size", type=str, default="L", choices=["S", "M", "L"],
                        help="SpectralX-S/M/L")
    parser.add_argument("--no_asb", action="store_true", default=False,
                        help="Disable Adaptive Spectral Block for ablation study")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=512, help="Batch size for testing")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train/val split ratio")
    
    # Data parameters
    parser.add_argument("--in_channels", type=int, default=52, help="Number of input channels (subcarriers)")
    parser.add_argument("--num_classes_act", type=int, default=6, help="Number of activity classes")
    parser.add_argument("--num_classes_loc", type=int, default=16, help="Number of location classes")
    parser.add_argument("--train_data", type=str, default="datasets/train_data_split_amp.mat", help="Training data path")
    parser.add_argument("--test_data", type=str, default="datasets/test_data_split_amp.mat", help="Test data path")
    
    # Loss weights
    parser.add_argument("--loss_weight_act", type=float, default=0.5, help="Weight for activity loss")
    parser.add_argument("--loss_weight_loc", type=float, default=0.5, help="Weight for location loss")
    
    # Device
    parser.add_argument("--cuda", type=str, default="1", help="GPU for training")
    
    return parser.parse_args()


def train(model, loss_fn, dataloader, optimizer, epoch, loss_weight_act, loss_weight_loc):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct_act, correct_loc = 0, 0
    
    for data, target_act, target_loc in dataloader:
        if torch.cuda.is_available():
            data = data.cuda()
            target_act = target_act.cuda()
            target_loc = target_loc.cuda()

        optimizer.zero_grad()
        output_act, output_loc, _ = model(data)
        
        # Compute losses
        loss_act = loss_fn(output_act, target_act)
        loss_loc = loss_fn(output_loc, target_loc)
        loss = loss_weight_act * loss_act + loss_weight_loc * loss_loc
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        
        # Activity accuracy
        pred_act = output_act.argmax(dim=1, keepdim=True)
        correct_act += pred_act.eq(target_act.view_as(pred_act)).sum().item()
        
        # Location accuracy
        pred_loc = output_loc.argmax(dim=1, keepdim=True)
        correct_loc += pred_loc.eq(target_loc.view_as(pred_loc)).sum().item()

    n = len(dataloader.dataset)
    print(f'Train Epoch: {epoch} \tLoss: {total_loss/n:.6f}, '
          f'Act Acc: {100.0*correct_act/n:.2f}%, Loc Acc: {100.0*correct_loc/n:.2f}%')


def evaluate(model, loss_fn, dataloader, epoch, loss_weight_act, loss_weight_loc):
    """Evaluate the model on validation set."""
    model.eval()
    total_loss = 0
    correct_act, correct_loc = 0, 0
    
    with torch.no_grad():
        for data, target_act, target_loc in dataloader:
            if torch.cuda.is_available():
                data = data.cuda()
                target_act = target_act.cuda()
                target_loc = target_loc.cuda()

            output_act, output_loc, _ = model(data)
            
            # Compute losses
            loss_act = loss_fn(output_act, target_act)
            loss_loc = loss_fn(output_loc, target_loc)
            loss = loss_weight_act * loss_act + loss_weight_loc * loss_loc
            
            total_loss += loss.item() * data.size(0)
            
            # Activity accuracy
            pred_act = output_act.argmax(dim=1, keepdim=True)
            correct_act += pred_act.eq(target_act.view_as(pred_act)).sum().item()
            
            # Location accuracy
            pred_loc = output_loc.argmax(dim=1, keepdim=True)
            correct_loc += pred_loc.eq(target_loc.view_as(pred_loc)).sum().item()

    n = len(dataloader.dataset)
    print(f'\nValidation: Loss: {total_loss/n:.4f}, '
          f'Act Acc: {100.0*correct_act/n:.2f}%, Loc Acc: {100.0*correct_loc/n:.2f}%\n')
    
    return total_loss / n, correct_act / n, correct_loc / n


def test(model, dataloader):
    """Test the model."""
    model.eval()
    correct_act, correct_loc = 0, 0
    
    with torch.no_grad():
        for data, target_act, target_loc in dataloader:
            if torch.cuda.is_available():
                data = data.cuda()
                target_act = target_act.cuda()
                target_loc = target_loc.cuda()

            output_act, output_loc, _ = model(data)
            
            # Activity accuracy
            pred_act = output_act.argmax(dim=1, keepdim=True)
            correct_act += pred_act.eq(target_act.view_as(pred_act)).sum().item()
            
            # Location accuracy
            pred_loc = output_loc.argmax(dim=1, keepdim=True)
            correct_loc += pred_loc.eq(target_loc.view_as(pred_loc)).sum().item()

    n = len(dataloader.dataset)
    acc_act = 100.0 * correct_act / n
    acc_loc = 100.0 * correct_loc / n
    
    print(f"Test Results:")
    print(f"  Activity Recognition Accuracy: {acc_act:.2f}%")
    print(f"  Indoor Localization Accuracy: {acc_loc:.2f}%")
    
    return acc_act, acc_loc


def train_and_evaluate(model, loss_fn, train_loader, val_loader, optimizer, scheduler,
                       epochs, save_path, loss_weight_act, loss_weight_loc):
    """Train and evaluate the model, saving the best model."""
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        train(model, loss_fn, train_loader, optimizer, epoch, loss_weight_act, loss_weight_loc)
        val_loss, val_acc_act, val_acc_loc = evaluate(model, loss_fn, val_loader, epoch, 
                                                       loss_weight_act, loss_weight_loc)
        scheduler.step()
        
        if val_loss < best_loss:
            print(f"Validation loss improved from {best_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
        else:
            print("Validation loss did not improve.")


def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.cuda
    setup_seed(conf.seed)
    os.makedirs("weights", exist_ok=True)
    
    # Determine whether to use ASB
    use_asb = not conf.no_asb
    
    # Generate save path
    save_path = get_save_path(conf.model_size, use_asb)
    model_name = f"SpectralX_{conf.model_size}" + ("" if use_asb else "_wo_ASB")
    
    # Print experiment configuration
    print("=" * 60)
    print("Multi-Task Learning: Activity Recognition + Indoor Localization")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Activity classes: {conf.num_classes_act}, Location classes: {conf.num_classes_loc}")
    print(f"Save path: {save_path}")
    print("=" * 60)

    # Training
    if conf.mode in ["train", "train_test"]:
        print("\nLoading training data...")
        x_train, y_train_act, y_train_loc, x_val, y_val_act, y_val_loc = TrainDataset(
            data_path=conf.train_data,
            train_ratio=conf.train_ratio,
            seed=conf.seed
        )
        
        print(f"Train set: {len(x_train)} samples")
        print(f"Val set: {len(x_val)} samples")
        print(f"Input shape: {x_train.shape[1:]} (channels, time)")
        
        train_loader = DataLoader(
            TensorDataset(x_train, y_train_act, y_train_loc),
            batch_size=conf.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val_act, y_val_loc),
            batch_size=conf.batch_size, shuffle=False
        )

        model = SpectralX(
            in_channels=conf.in_channels,
            num_classes_act=conf.num_classes_act,
            num_classes_loc=conf.num_classes_loc,
            nf=get_param_value(conf.model_size),
            use_asb=use_asb
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
            gamma=0.5
        )
        loss_fn = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.cuda()
            loss_fn = loss_fn.cuda()
            summary(model, (conf.in_channels, 192))

        print("\nStarting training...")
        train_and_evaluate(model, loss_fn, train_loader, val_loader, optimizer, scheduler,
                          conf.epochs, save_path, conf.loss_weight_act, conf.loss_weight_loc)

    # Testing
    if conf.mode in ["test", "train_test"]:
        print("\nLoading test data...")
        x_test, y_test_act, y_test_loc = TestDataset(
            data_path=conf.test_data,
            seed=conf.seed
        )
        
        print(f"Test set: {len(x_test)} samples")
        
        test_loader = DataLoader(
            TensorDataset(x_test, y_test_act, y_test_loc),
            batch_size=conf.test_batch_size, shuffle=False
        )

        model = SpectralX(
            in_channels=conf.in_channels,
            num_classes_act=conf.num_classes_act,
            num_classes_loc=conf.num_classes_loc,
            nf=get_param_value(conf.model_size),
            use_asb=use_asb
        )
        if torch.cuda.is_available():
            model = model.cuda()
        model.load_state_dict(torch.load(save_path, weights_only=True))

        print(f"\nTesting {model_name}...")
        acc_act, acc_loc = test(model, test_loader)
        
        return acc_act, acc_loc


if __name__ == "__main__":
    main()
