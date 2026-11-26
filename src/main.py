from pathlib import Path
from typing import Optional
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger

from src.data import ArtworkDataModule
from src.training import ArtworkClassifier


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_name", type=str, default="run")
        parser.add_argument("--out_dir", type=str, default="runs")
        parser.add_argument("--wandb_project", type=Optional[str], default=None)
        parser.add_argument("--wandb_entity", type=Optional[str], default=None)

        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.img_size", "model.img_size")
        parser.link_arguments("data.classes", "model.classes", apply_on="instantiate")

    def before_instantiate_classes(self):
        #check that exists checkpoint path for validate/test
        if self.subcommand in ("validate", "test") and not self.config[self.subcommand].ckpt_path:
            raise ValueError(f"{self.subcommand} needs --ckpt_path to load the trained model")

    def instantiate_trainer(self, **kwargs):
        config = self.config_init[self.subcommand]
        run_folder = str(Path(config.out_dir) / config.run_name)

        callbacks = list(config.trainer.callbacks or [])
        # best.ckpt = best val macro-F1, last.ckpt = last epoch (save_last)
        callbacks.append(ModelCheckpoint(dirpath=run_folder + "/checkpoints", filename="best", monitor="val/macro_f1",
                                         mode="max", save_top_k=1, save_last=True, enable_version_counter=False,
                                         save_on_train_epoch_end=True))

        logger = False
        if config.wandb_project:
            # same id as the run name, so resuming continues the same wandb run
            # log_model=True: best.ckpt and last.ckpt are uploaded as artifacts at the end of fit
            logger = WandbLogger(project=config.wandb_project, entity=config.wandb_entity, name=config.run_name,
                                 id=config.run_name, resume="allow", save_dir=run_folder, log_model=True)
        return super().instantiate_trainer(default_root_dir=run_folder, callbacks=callbacks, logger=logger, **kwargs)


def main():
    CLI(
        ArtworkClassifier,
        ArtworkDataModule,
        save_config_callback=None,
        trainer_defaults={"num_sanity_val_steps": 0, "enable_model_summary": False},
    )


if __name__ == "__main__":
    main()
