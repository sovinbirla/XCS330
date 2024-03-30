import argparse
import os
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
from collections import defaultdict

import utils
import transformers


parser = argparse.ArgumentParser()
parser.add_argument('--task')
parser.add_argument('--model')
parser.add_argument('--dataset')
parser.add_argument('--k', default='0')
parser.add_argument('--mode', default='all')
parser.add_argument('--prompt', default='qa')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--repeats', default=1, type=int)
parser.add_argument('--output', default='plot.png')
parser.add_argument('--device', default='gpu')
args = parser.parse_args()

os.environ["DEVICE"] = args.device

# Check if submission module is present.  If it is not, then main() will not be executed.
use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
    import submission

def plot():
    dataset = 'xsum'
    data = defaultdict(lambda: defaultdict(list))
    model = 'med'
    mode = 'lora16'
    x_vals = set()
    for k in [0,1,8,128]:
        fn = '_'.join([model, dataset, str(k), mode])
        id_ = '_'.join([model, dataset, mode])
        with open(f'submission/results/ft/{fn}.json', 'r') as f:
            score = json.load(f)['metric']
            data[id_]['x'].append(k)
            x_vals.add(k)
            data[id_]['y'].append(score)
    
    prompt_mode = 'tldr'
    for k in [0,1,4]:
        fn = '_'.join([model, dataset, str(k), prompt_mode])
        id_ = '_'.join([model, dataset, prompt_mode])
        with open(f'submission/results/icl/{fn}.json', 'r') as f:
            score = json.load(f)['metric']
            data[id_]['x'].append(k)
            x_vals.add(k)
            data[id_]['y'].append(score)

    for k, v in data.items():
        plt.plot(v['x'], v['y'], label=k)
    if max(x_vals) > 4:
        plt.xscale('symlog')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(sorted(x_vals))
    plt.legend()
    plt.title(dataset)
    plt.ylabel(utils.metric_for_dataset(dataset))
    plt.xlabel('Number of support examples')
    plt.savefig(args.output, bbox_inches='tight')

# Download all models and datasets required by the grader
def cache():

    models = [
        {'name': 'bert-tiny', 'type':  transformers.AutoModelForSequenceClassification, 'num_labels': 5},
        {'name': 'bert-tiny', 'type':  transformers.AutoModelForCausalLM},
        {'name': 'med', 'type':  transformers.AutoModelForCausalLM}
    ]
    
    for model in models:
        if 'num_labels' in model:
            utils.get_model_and_tokenizer(model['name'], model['type'], num_labels = model['num_labels'])
        else:
            utils.get_model_and_tokenizer(model['name'], model['type'])

    
    datasets = [
        {'name': 'amazon', 'n_train': 1, 'n_val': 125},
        {'name': 'xsum', 'n_train': 8, 'n_val': 125}
    ]

    for dataset in datasets:
        utils.get_dataset(dataset=dataset['name'], n_train=dataset['n_train'], n_val=dataset['n_val'])

def run():
    ks = [int(k) for k in args.k.split(',')]
    if args.task == 'run_ft':
        submission.run_ft(args.model.split(','), args.dataset.split(','), ks, args.mode.split(','), args.debug, args.repeats)
    elif args.task == 'run_icl':
        submission.run_icl(args.model.split(','), args.dataset.split(','), ks, args.prompt.split(','), args.debug, args.repeats)
    elif args.task == 'plot_ft':
        submission.plot_ft(args.model.split(','), args.dataset.split(','), ks, args.mode.split(','), args.output)
    elif args.task == 'plot_icl':
        assert ',' not in args.dataset, "Only one dataset at a time for plotting"
        submission.plot_icl(args.model.split(','), args.dataset, ks, args.prompt.split(','), args.output)
    elif args.task == 'plot':
        plot()
    elif args.task == 'cache':
        cache()

if __name__ == '__main__':
    run()