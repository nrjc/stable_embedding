# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gin


def run_main():
    # Use a breakpoint in the code line below to debug your script.
    gin.parse_config_file('config.gin')

    trainer = get_trainer()
    trainer()


@gin.configurable
def get_trainer(trainer=gin.REQUIRED):
    return trainer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
