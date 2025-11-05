import random
import gridsearch
from importlib.resources import files
from pathlib import Path
random.seed(9100)
filein_hp = "hyperparameters_test.json"


print("beginning optimizer test routine")
actual_path = Path(files("gridsearch.data").joinpath(filein_hp))
hpoptimizer = gridsearch.HPoptimizer(actual_path, step_carefully=False)

optimalidxs = {}
for key in hpoptimizer.keys:
    dim = len(hpoptimizer.parameterspacebykey[key])
    optimalidxs[key] = random.randint(0, dim - 1)
    hpoptimizer.current_idx[key] = random.randint(0, dim - 1)

print("randomly selected optimal coordinate:")
for key in optimalidxs.keys():
    print("", key, hpoptimizer.parameterspacebykey[key][optimalidxs[key]])
print()
print("randomly selected starting coordinate:")
for key in optimalidxs.keys():
    print("", key, hpoptimizer.parameterspacebykey[key][hpoptimizer.current_idx[key]])
print()


def getscore(testcoordinate, optimalidxs):
    score = 0
    for key in testcoordinate.keys():
        score += abs(testcoordinate[key] - optimalidxs[key])
    return score


while not hpoptimizer.optimized:
    neighbouring_unscored_coordinate = hpoptimizer.get_next_neighbouring_unscored_coordinate()

    while neighbouring_unscored_coordinate is not None:
        # parameters = hpoptimizer.get_parameters(neighbouring_unscored_coordinate)

        score = getscore(neighbouring_unscored_coordinate, optimalidxs)

        hpoptimizer.update_score(neighbouring_unscored_coordinate, score)
        neighbouring_unscored_coordinate = hpoptimizer.get_next_neighbouring_unscored_coordinate()

    hpoptimizer.step()


print("ending coordinate, score:")
print(hpoptimizer.current_idx)
print(hpoptimizer.get_score(hpoptimizer.current_idx))

success = True
for key in optimalidxs.keys():
    print("", key, hpoptimizer.parameterspacebykey[key][optimalidxs[key]])
    success = success and optimalidxs[key] == hpoptimizer.current_idx[key]
if success:
    print(f"SUCCESS in {hpoptimizer.steps} steps")
else:
    print(f"FAILURE in {hpoptimizer.steps} steps")
