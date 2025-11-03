import json
import numpy as np
import sys
import copy

class HPoptimizer:
    def __init__(self, filein_hp, step_carefully = False):
        self.filein_hp = filein_hp
        self.step_carefully = step_carefully
        self.update_file_clear_stepping() #clear previous optimization info

        #define hyperparameters:
        with open(self.filein_hp) as fi:
            jsoncontent = json.load(fi)
            hpidx0 = jsoncontent['hyperparameter_idx0']
            self.hp_allow = jsoncontent['hyperparameter_allowmove']
            hpopt = jsoncontent['hyperparameter_optimisation']

        self.current_idx = {}
        self.parameterspacebykey = {}
        self.dims = []
        self.keys = []
        for key in hpopt.keys():
            if self.hp_allow[key]:
                self.current_idx[key] = hpidx0[key]
                self.parameterspacebykey[key] = hpopt[key]
                self.dims.append(len(hpopt[key]))
            else: # restrict the hyperparameter to its current idx0 value:
                self.current_idx[key] = 0
                self.parameterspacebykey[key] = [hpopt[key][hpidx0[key]]]
                self.dims.append(1)


            self.keys.append(key)

        self.scores = np.ones(self.dims) * -1
        self.steps = 0

        self.optimized = False

    def get_neighbouring_coordinates(self):
        #return a list containing the current coordinate plus neighbouring (current + single perturbation)
        perturbations = [self.current_idx]

        #get combinations of surrounding coordinates in hyperparameter space:
        for key_perturbed in self.keys:
            if self.current_idx[key_perturbed] < len(self.parameterspacebykey[key_perturbed]) - 1:
                perturbed_idx = {}
                for key_unperturbed in [x for x in self.keys if x != key_perturbed]:
                    perturbed_idx[key_unperturbed] = self.current_idx[key_unperturbed]

                perturbed_idx[key_perturbed] = self.current_idx[key_perturbed] + 1
                perturbations.append(perturbed_idx)

            if self.current_idx[key_perturbed] > 0:
                perturbed_idx = {}
                for key_unperturbed in [x for x in self.keys if x != key_perturbed]:
                    perturbed_idx[key_unperturbed] = self.current_idx[key_unperturbed]

                perturbed_idx[key_perturbed] = self.current_idx[key_perturbed] - 1
                perturbations.append(perturbed_idx)

        return perturbations

    def get_next_neighbouring_unscored_coordinate(self):
        neighbouring = self.get_neighbouring_coordinates()

        next_perturbation = None
        for perturbed_idx in neighbouring:
            # check to see if unscored:
            if self.get_score(perturbed_idx) == -1:
                next_perturbation = perturbed_idx
                break

        return next_perturbation

    def get_parameters(self, coordinate):
        values = {}
        for key in self.keys:
            #values.append(self.parameterspacebykey[key][self.current_idx[key]])
            values[key] = self.parameterspacebykey[key][coordinate[key]]
        return values

    def update_score(self, coordinate, score):
        slc = []
        for dim, key in enumerate(self.keys):
            #slc.append(slice(self.current_idx[key], self.current_idx[key]+1))
            slc.append(slice(coordinate[key], coordinate[key] + 1))
        slc = tuple(slc)
        self.scores[slc] = score

    def get_score(self, coordinate):
        slc = []
        for dim, key in enumerate(self.keys):
            #slc.append(slice(self.current_idx[key], self.current_idx[key]+1))
            slc.append(slice(coordinate[key], coordinate[key] + 1))
        slc = tuple(slc)
        score= self.scores[slc].reshape(-1)[0]
        return score

    def step(self):
        neighbouring = self.get_neighbouring_coordinates()

        if self.get_next_neighbouring_unscored_coordinate() is not None:
            print("warning: not all neighbouring coordinates have been scored before attempting to step optimizer")

        successful_perturbation_directions = {}
        successful_perturbation_scorereductions = {}

        #store the score of the current coordinate:
        scores = [self.get_score(self.current_idx)]
        score_min = scores[0]
        score_min_idx = 0
        print("evaluating scores of nearby hyperparameter coordinates:")
        for idx, perturbed_idx in enumerate(neighbouring): #skip the coordinate that we are already on
            score = self.get_score(perturbed_idx)
            print("",idx, score, perturbed_idx)

            if idx == 0: continue

            if score < score_min and score != -1:
                score_min = score
                score_min_idx = idx

            if score < scores[0]:
                #find which coordinate was perturbed to achieve a lower score:
                perturbed_key = [key for key in self.keys if perturbed_idx[key] != self.current_idx[key]][0]
                direction = perturbed_idx[perturbed_key] - self.current_idx[perturbed_key]
                scorereduction = scores[0] - score
                if perturbed_key in successful_perturbation_directions.keys():
                    if successful_perturbation_scorereductions[perturbed_key] < scorereduction:
                        successful_perturbation_directions[perturbed_key] = direction
                        successful_perturbation_scorereductions[perturbed_key] = scorereduction
                else:
                    successful_perturbation_directions[perturbed_key] = direction
                    successful_perturbation_scorereductions[perturbed_key] = scorereduction

            scores.append(score)

        if score_min_idx == -1:
            print("error: could not find any scores to evaluate during step")
            sys.exit(1)
        elif score_min_idx == 0:
            #the current coordinate has the lowest score, so finish
            print("current index is lowest score, not stepping")
            print()
            self.optimized = True
            return False
        else:
            #move the optimizer to a new coordinate
            if self.step_carefully:
                best_move = copy.deepcopy(neighbouring[score_min_idx])
                score_new = score_min
            else:
                best_move = copy.deepcopy(self.current_idx)
                for key in self.keys:
                    if key in successful_perturbation_directions.keys():
                        best_move[key] = best_move[key] + successful_perturbation_directions[key]
                #if score at best_move, use it:
                score_new = self.get_score(best_move)

                #if we have not stepped here before, score_new = -1
                if score_new != -1:
                    #otherwise, check that this actually reduces the score:
                    if score_new >= score_min:
                        #then do a careful step instead:
                        best_move = copy.deepcopy(neighbouring[score_min_idx])
                        score_new = score_min

            print("stepping from:")
            print("",self.current_idx)
            print("","to :")
            print("",best_move)
            print()
            for key in self.keys:
                self.current_idx[key] = best_move[key]

            self.steps = self.steps + 1
            self.update_file_with_score(self.current_idx, score_new)

            # update default coordinate in file to the newly optimised coordinate:
            self.update_file_with_new_default_coordinate(self.current_idx)
            return True

    def update_file_with_score(self, best_idx, score):
        print(f"updating {self.filein_hp} with new score")

        #preserve idx0 for parameters which shouldn't be moved:
        best_idx_parameterspace = {}
        for key in best_idx:
            if self.hp_allow[key]:
                best_idx_parameterspace[key] = best_idx[key]
            else:
                best_idx_parameterspace[key] = self.hpidx0[key]

        with open(self.filein_hp) as fi:
            jsoncontent = json.load(fi)

        step = {"score": score,
                "hyperparameter set": best_idx_parameterspace}

        jsoncontent["stepping"][self.steps] = step

        json_object = json.dumps(jsoncontent, indent=4)
        with open(self.filein_hp, 'w') as fo:
            fo.write(json_object)

    def update_file_with_new_default_coordinate(self, best_idx):
        print(f"updating {self.filein_hp} with new default coordinate")

        #preserve idx0 for parameters which shouldn't be moved:
        best_idx_parameterspace = {}
        for key in best_idx:
            if self.hp_allow[key]:
                best_idx_parameterspace[key] = best_idx[key]
            else:
                best_idx_parameterspace[key] = self.hpidx0[key]

        with open(self.filein_hp) as fi:
            jsoncontent = json.load(fi)

        #overwrite previous idx0
        jsoncontent["hyperparameter_idx0"] = best_idx_parameterspace

        json_object = json.dumps(jsoncontent, indent=4)
        with open(self.filein_hp, 'w') as fo:
            fo.write(json_object)

    def update_file_clear_stepping(self):
        with open(self.filein_hp) as fi:
            jsoncontent = json.load(fi)

        jsoncontent["stepping"] = {}

        json_object = json.dumps(jsoncontent, indent=4)
        with open(self.filein_hp, 'w') as fo:
            fo.write(json_object)

    def print_current_parameters(self):
        print("hyperparameters:")
        for key in self.keys:
            print("", key, self.parameterspacebykey[key][self.current_idx[key]])
        print()
