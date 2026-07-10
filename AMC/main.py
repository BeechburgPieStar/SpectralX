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
    parser = argparse.ArgumentParser(description="SpectralX Training and Testing")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "train_test"],
                        help="Choose mode: 'train', 'test', or 'train_test'.")
    parser.add_argument("--model_size", type=str, default="S", help="SpectralX-S/M/L")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--num_classes", type=int, default=24, help="Number of classes")
    parser.add_argument("--cuda", type=str, default="0", help="GPU for training")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--use_asb", action="store_true", default=True,
                        help="Use Adaptive Spectral Block (default: True)")
    parser.add_argument("--no_asb", action="store_true", default=False,
                        help="Disable Adaptive Spectral Block for ablation study")
    return parser.parse_args()


def train(model, loss_fn, dataloader, optimizer, epoch):
    """Train the model for one epoch."""
    model.train()
    total_loss, correct = 0, 0
    for data, target in dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        total_loss / len(dataloader.dataset),
        correct,
        len(dataloader.dataset),
        100.0 * correct / len(dataloader.dataset))
    )


def evaluate(model, loss_fn, dataloader, epoch):
    """Evaluate the model on validation set."""
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            total_loss += loss_fn(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            total_loss / len(dataloader.dataset),
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
        )
    )
    return total_loss / len(dataloader.dataset)


def test(model, dataloader):
    """Test the model."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return correct / len(dataloader.dataset)


def train_and_evaluate(model, loss_fn, train_loader, val_loader, optimizer, epochs, save_path, patience=5):
    """Train and evaluate the model, saving the best model with early stopping."""
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train(model, loss_fn, train_loader, optimizer, epoch)
        val_loss = evaluate(model, loss_fn, val_loader, epoch)

        if val_loss < best_loss or epoch == 1:
            print(f"The validation loss is improved from {best_loss:.4f} to {val_loss:.4f}, new model weight is saved.")
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
        else:
            print("The validation loss is not improved.")
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping: No improvement for {patience} epochs. Training stopped.")
            break


def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.cuda
    setup_seed(conf.seed)
    
    # Determine whether to use ASB (--no_asb overrides --use_asb)
    use_asb = not conf.no_asb
    
    # Generate save path based on model config
    save_path = get_save_path(conf.model_size, use_asb)
    model_name = f"SpectralX_{conf.model_size}" + ("" if use_asb else "_wo_ASB")
    print(f"Model: {model_name}")
    print(f"Save path: {save_path}")

    # Training
    if conf.mode in ["train", "train_test"]:
        x_train, y_train, x_val, y_val = TrainDataset(conf.seed)
        train_loader = DataLoader(TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)), 
                                batch_size=conf.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val)), 
                                batch_size=conf.batch_size, shuffle=True)

        model = SpectralX(num_classes=conf.num_classes, nf=get_param_value(conf.model_size), use_asb=use_asb)
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
        loss_fn = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.cuda()
            loss_fn = loss_fn.cuda()
            summary(model, (2, 128))

        print("Starting training...")
        train_and_evaluate(model, loss_fn, train_loader, val_loader, optimizer, conf.epochs, save_path, conf.patience)

    # Testing
    if conf.mode in ["test", "train_test"]:
        model = SpectralX(num_classes=conf.num_classes, nf=get_param_value(conf.model_size), use_asb=use_asb)
        if torch.cuda.is_available():
            model = model.cuda()
        model.load_state_dict(torch.load(save_path, weights_only=True))

        print(f"\nModel: {model_name}")
        test_results = {}
        for test_snr in range(-20, 32, 2):
            x_test, y_test = TestDataset(test_snr, conf.seed)
            test_loader = DataLoader(TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), 
                                     batch_size=conf.test_batch_size, shuffle=False)
            acc = test(model, test_loader)
            test_results[test_snr] = acc
            print(f'Test Accuracy on {test_snr} dB = {acc:.4f}')

        return test_results


if __name__ == "__main__":
    main()
