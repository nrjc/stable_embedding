# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gin

from trainer.VanillaTrainer import VanillaTrainer
from trainer.pinned_trainer import PinnedTrainer


def run_main():
    # Use a breakpoint in the code line below to debug your script.
    gin.parse_config_file('config.gin')

    trainer = get_vanilla_trainer()
    trainer()
    trainer.run_all_and_save()
    pinned_trainer = get_pinned_trainer()
    pinned_trainer()
    pinned_trainer.evaluate()


@gin.configurable
def get_vanilla_trainer(trainer=gin.REQUIRED) -> VanillaTrainer:
    return trainer


@gin.configurable
def get_pinned_trainer(trainer=gin.REQUIRED) -> PinnedTrainer:
    return trainer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
