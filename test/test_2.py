import random
import gridsearch
from importlib.resources import files
from pathlib import Path
random.seed(9100)
filein_hp = "hyperparameters_test.json"


print("beginning optimizer test routine")
actual_path = Path(files("gridsearch.data").joinpath(filein_hp))


#hpoptimizer = gridsearch.HPoptimizer(actual_path, step_carefully=False)
searcher = gridsearch.Searcher(actual_path, step_carefully=False)

#set an arbitrary optimal index:
optimalidxs = {}
for key in searcher.optimizer.keys:
    dim = len(searcher.optimizer.parameterspacebykey[key])
    optimalidxs[key] = random.randint(0, dim - 1)
    searcher.optimizer.current_idx[key] = random.randint(0, dim - 1)

print("randomly selected optimal coordinate:")
for key in optimalidxs.keys():
    print("", key, searcher.optimizer.parameterspacebykey[key][optimalidxs[key]])
print()
print("randomly selected starting coordinate:")
for key in optimalidxs.keys():
    print("", key, searcher.optimizer.parameterspacebykey[key][searcher.optimizer.current_idx[key]])
print()


def getscore(testcoordinate, optimalidxs):
    score = 0
    for key in testcoordinate.keys():
        score += abs(testcoordinate[key] - optimalidxs[key])
    return score


counter = 0
for coordinate in searcher:
    score = getscore(coordinate.get_idx(), optimalidxs)
    coordinate.update(score)
    counter += 1

pdict, score = searcher.get_optimal_pdict()
print(" {} iterations used".format(counter))
print("ending coordinate, score:")
print(pdict)
print(score)

success = True
for key in optimalidxs.keys():
    print("", key, searcher.optimizer.parameterspacebykey[key][optimalidxs[key]])
    success = success and optimalidxs[key] == searcher.optimizer.current_idx[key]
if success:
    print(f"SUCCESS in {searcher.optimizer.steps} steps")
else:
    print(f"FAILURE in {searcher.optimizer.steps} steps")
