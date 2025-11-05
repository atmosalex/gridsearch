import numpy as np
import gridsearch
import sys

if __name__ == '__main__':
    fileout_hp = "hyperparameters_test.json"

    gridsearch.test_optimizer(fileout_hp)

    sys.exit()

    # optimize hyperparameters:
    hpoptimizer = gridsearch.HPoptimizer(fileout_hp)  # this file might be modified if HPs are optimized

    # optimize hyperparameters for each model optimization
    while not hpoptimizer.optimized:
        neighbouring_unscored_coordinate = hpoptimizer.get_next_neighbouring_unscored_coordinate()
        print("beginning new round of hyperparameter optimization")
        print("#")
        while neighbouring_unscored_coordinate is not None:
            info_model = hpoptimizer.get_parameters(neighbouring_unscored_coordinate)


            hpoptimizer.update_score(neighbouring_unscored_coordinate, np.mean(np.array(scores_each_tv_test_split)))
            neighbouring_unscored_coordinate = hpoptimizer.get_next_neighbouring_unscored_coordinate()

        hpoptimizer.step()
        print()

    # train model with current set of hyperparameters:
    info_model = hpoptimizer.get_parameters(hpoptimizer.current_idx)

