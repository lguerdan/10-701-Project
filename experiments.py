from train import *
import helpers
import copy


def run_CL_experiment(params, clipping_methods, cl_scenario):
    for method in clipping_methods:
        exp_name = f'{cl_scenario}_CL_adaptive_cuttoff__C_{method}'
        print('\n############################')
        print(f'Running setting: {exp_name}')
        print('############################\n')
        exp_params = copy.copy(params)
        exp_params['clipping'] = method
        run_continual_exp(exp_name, exp_params, use_devset=exp_params['use_devset'], cl_scenario=cl_scenario)

def run_central_baseline_fixed_cuttoff(params, cutoffs):
    for S in cutoffs:
        exp_name = f'central_baseline_fixed_cutoff__S_{S}'

        print('\n############################')
        print(f'Running setting: {exp_name}')
        print('############################\n')
        exp_params = copy.copy(params)
        exp_params['S'] = S
        run_exp(exp_name, exp_params, use_devset=exp_params['use_devset'])


def run_central_baseline_adaptive_cuttoff(params, clipping_methods):
    for method in clipping_methods:
        exp_name = f'central_baseline_adaptive_cuttoff__C_{method}'

        print('\n############################')
        print(f'Running setting: {exp_name}')
        print('############################\n')
        exp_params = copy.copy(params)
        exp_params['clipping'] = method
        run_exp(exp_name, exp_params, use_devset=exp_params['use_devset'])


if __name__ == "__main__":

    # Experiment 1 configuration - vary fixed threshold (S) under centralized evaluation
    exp1_params = {
        'use_devset': False,
        'lr': 0.05,
        'dp': True,
        'clipping': 'Fixed',
        'num_microbatches': 32,
        'batch_size': 32,
        'S': 1,
        'z': 0.2,
        'gamma': 0.5,
        'lr_c': 0.2,
        'momentum': 0.5,
        'decay': 0,
        'n_epochs': 60,
    }

    
    # Experiment 2 configuration - vary threshold update type under centralized training
    exp2_params = {
        'use_devset': False,
        'lr': 0.05,
        'dp': True,
        'clipping': 'Linear',
        'num_microbatches': 32,
        'batch_size': 32,
        'S': 1,
        'z': 0.2,
        'gamma': 0.7,
        'lr_c': 0.1,
        'momentum': 0.5,
        'decay': 0,
        'n_epochs': 60,
    }

    # Experiment 3 configuration - vary threshold update type under continual learning scenarios
    exp3_params = {
        'use_devset': True,
        'lr': 0.05,
        'dp': True,
        'clipping': 'Linear',
        'num_microbatches': 32,
        'batch_size': 32,
        'S': 5,
        'z': 0.1,
        'gamma': 0.7,
        'lr_c': 0.01,
        'momentum': 0.5,
        'decay': 0,
        'n_epochs': 6,
    }

    cutoffs = [.5, 1, 1.5, 2]
    clipping_methods = ['Fixed', 'Linear', 'Exponential']

    # run_central_baseline_fixed_cuttoff(exp1_params, cutoffs)
    # run_central_baseline_adaptive_cuttoff(exp2_params, clipping_methods)
    run_CL_experiment(exp3_params, clipping_methods, cl_scenario='Class')
    # run_CL_experiment(exp3_params, clipping_methods, cl_scenario='Instance')
