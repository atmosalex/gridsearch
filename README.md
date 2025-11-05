## `gridsearch`:

a simple-to-use grid search implementation in Python, designed to easily integrate into a model training loop

## Installation:

`git clone https://github.com/atmosalex/gridsearch.git`
`cd gridsearch`
`python -m build`
`pip install .`

## Usage:

1. write a .json file defining hyperparameter space, in the style of the provided example at `src/gridsearch/data/hyperparameters_test.json`

2. create an instance of `gridsearch.Searcher` then iterate over each hyperspace coordinate as shown below, calling the `update` method until the `Searcher` determines the optimal coordinate:

```
searcher = gridsearch.Searcher('hyperparameters_file.json', step_carefully=False)

for coordinate in searcher:
	dict_hyperparams = coordinate.get_pdict()
    score = train_model(dict_hyperparams)
    coordinate.update(score)
```
