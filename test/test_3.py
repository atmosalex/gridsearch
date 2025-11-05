import gridsearch
from importlib.resources import files
from pathlib import Path
filein_hp = "hyperparameters_test.json"
actual_path = Path(files("gridsearch.data").joinpath(filein_hp))

hp0 = gridsearch.load_default_hyperparameters(actual_path)

print(hp0)