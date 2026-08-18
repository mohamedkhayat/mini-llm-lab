import hydra
import omegaconf
from training.trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))
    print("-" * 60)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
