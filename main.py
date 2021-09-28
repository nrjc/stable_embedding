# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gin

from trainer.Trainer import Trainer


def run_main():
    # Use a breakpoint in the code line below to debug your script.
    gin.parse_config_file('config.gin')

    trainer = Trainer()
    trainer()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
