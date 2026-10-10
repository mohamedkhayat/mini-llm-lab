import hydra
import omegaconf

from data.cache import prepare_cache_for_run
from training.assembly import build_trainer
from training.checkpointing import load_checkpoint, resolve_resume_path


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    resume_path = resolve_resume_path(cfg.training)
    checkpoint = load_checkpoint(resume_path) if resume_path is not None else None
    prepare_cache_for_run(cfg, checkpoint)

    print(omegaconf.OmegaConf.to_yaml(cfg))
    print("-" * 60)

    trainer = build_trainer(cfg, checkpoint=checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
