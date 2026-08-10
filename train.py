import hydra
import omegaconf
from data.dataloader import create_dataloaders


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))
    print("-" * 60)

    train_loader, val_loader = create_dataloaders(cfg.data)

    x, y = next(iter(train_loader))
    print(f"Train batch: x={x.shape}, y={y.shape}, dtype={x.dtype}")
    print(f"Steps/epoch (train): {len(train_loader)}, (val): {len(val_loader)}")


if __name__ == "__main__":
    main()
