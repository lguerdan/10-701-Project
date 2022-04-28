import json
import pandas as pd
import glob

def write_logs(exp_name, log, log_type, params=None):
    # Directly from dictionary
    df = pd.DataFrame.from_dict(log)
    df.to_csv(f'runs/{exp_name}_{log_type}.csv', index=False)

    if params:
        with open(f'runs/{exp_name}_params.json', "w") as outfile:
            json.dump(params, outfile)

def load_run(run_name):
    
    mnist, cifar  = pd.read_csv(f'runs/{run_name}/mnist.csv'), pd.read_csv(f'runs/{run_name}/cifar.csv')
    mnist['benchmark'] = 'MNIST'
    cifar['benchmark'] = 'CIFAR'
    log_data = pd.concat([mnist, cifar])

    with open(f'runs/{run_name}/mnist_params.json') as json_file:
        params = json.load(json_file)

    return log_data, params

def load_exp(exp_name):
    '''
        exp_name: stem of experiment name (e.g., baseline_experiment*)
        returns: dataframe over all results, param configuration
    '''

    exp_dfs = []
    for exp in glob.glob(f'runs/{exp_name}*'):
        exp_var = exp.split('__')[1].split('_')[0]
        exp_val = exp.split('__')[1].split('_')[1].split('/')[0]
        
        mnist, cifar  = pd.read_csv(f'{exp}/mnist.csv'), pd.read_csv(f'{exp}/cifar.csv')
        mnist['benchmark'] = 'MNIST'
        cifar['benchmark'] = 'CIFAR'
        log_data = pd.concat([mnist, cifar])
        log_data[exp_var] = exp_val
        
        exp_dfs.append(log_data)

    param_path = glob.glob(f'runs/{exp_name}*')[0]
    with open(f'{param_path}/mnist_params.json') as json_file:
        exp_params = json.load(json_file)
    exp_results = pd.concat(exp_dfs)

    return pd.concat(exp_dfs), exp_params
