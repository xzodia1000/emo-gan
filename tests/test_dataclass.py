from hparams import create_hparams
from model_utils import prepare_dataloaders


def test_prepare_dataloaders():
    hparams = create_hparams()
    train_loader, mixed_train_loader, val_loader, mixed_val_loader = (
        prepare_dataloaders(hparams)
    )

    print(f"train_loader: {len(train_loader)}")
    print(f"mixed_train_loader: {len(mixed_train_loader)}")
    print(f"val_loader: {len(val_loader)}")
    print(f"mixed_val_loader: {len(mixed_val_loader)}")


test_prepare_dataloaders()
