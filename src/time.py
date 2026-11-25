
from pathlib import Path

import pandas as pd
from tabulate import tabulate
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "params.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text())


NUM_EVOLVE = 1
NUM_EPOCH = config["train"]["epochs"]
# NUM_EPOCH = 10

TIME_PER_EPOCH = 11


def main():

    seconds = NUM_EVOLVE * TIME_PER_EPOCH * NUM_EPOCH

    minuts = seconds / 60
    hours = minuts / 60
    days = hours / 24

    rows = [{
            "minuts": f"{minuts:.0f}",
            "hours": f"{hours:.1f}",
            "days": f"{days:.2f}",
        }]

    rows = pd.DataFrame(rows)

    print(tabulate(rows, headers=list(rows), tablefmt="rounded_grid"), "\n")

if __name__ == "__main__":
    main()
