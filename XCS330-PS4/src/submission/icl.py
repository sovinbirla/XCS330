from cProfile import label
from multiprocessing.sharedctypes import Value
from typing import Dict, List, Optional, Tuple
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import transformers
import numpy as np
import random

import argparse
from collections import defaultdict
import json
import os
from rouge_score import rouge_scorer
import tqdm

import utils
import os
import glob

# DEVICE = os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu"
DEVICE = os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu"

# if (
#     DEVICE == "gpu"
#     and torch.backends.mps.is_available()
#     and torch.backends.mps.is_built()
# ):
#     DEVICE = torch.device("mps")
# el
if DEVICE == "gpu" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

    DEVICE = torch.device("mps")

print("In-context learning using device: ", DEVICE)


def get_icl_prompts(
    support_inputs: List[str],
    support_labels: List[str],
    test_input: str,
    prompt_mode: str = "qa",
) -> str:
    """
    Take a list of contexts and combine them into k-shot prompts.

    **Note**: Be sure to shuffle the support examples and labels
      *together* (i.e. so the pairings of support input/label is preserved)
      before constructing the prompt. np.random.permutation may be helpful.

    Args:
      support_inputs: The k inputs used for in-context learning (k may be zero!)
      support_labels: The k labels used for in-context learning (k may be zero!)
      test_input: The input we are evaluating on
      prompt_mode: The task description mode we're using; 'none' means we're only using
        k-shot examples, 'tl;dr' means we're using the tl;dr prompt from the GPT-2 paper,
        'qa' means we are adding "In the" after the question and before the answer and
        'custom' means your own prompt format for article summarization
        **different from the all the prompt formats previously mentioned**

    Returns:
      A string containing the complete input to the model.
    """

    prompt = ""

    ### START CODE HERE ###
    # Shuffle support examples and labels together
    combined = list(zip(support_inputs, support_labels))
    np.random.shuffle(combined)
    support_inputs, support_labels = zip(*combined)

    if prompt_mode == "qa":
        for input, label in zip(support_inputs, support_labels):
            prompt += f"{input} In the {label}. "
        prompt += f"{test_input} In the"
    elif prompt_mode == "none":
        for input, label in zip(support_inputs, support_labels):
            prompt += f"{input} {label} "
        prompt += f"{test_input}"
    elif prompt_mode == "tldr":
        for input, label in zip(support_inputs, support_labels):
            prompt += f"{input} TL;DR: {label}. "
        prompt += f"{test_input} TL;DR:"
    elif prompt_mode == "custom":
        # Example custom prompt for XSum (feel free to modify)
        for input, label in zip(support_inputs, support_labels):
            prompt += f"Summary of the article: {input} is {label}. "
        prompt += f"Summary of the article: {test_input} is"
    return prompt.strip()


def get_performance_metric(
    predictions: List[str], targets: List[str], metric: str
) -> float:
    if metric == "rouge":
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = []
        for p, t in zip(predictions, targets):
            score = scorer.score(p, t)["rouge1"].fmeasure
            scores.append(score)
        return sum(scores) / len(scores)
    elif metric == "exact match":
        if isinstance(targets[0], str):
            return sum(
                [p.strip() == t.strip() for p, t in zip(predictions, targets)]
            ) / len(predictions)
        else:

            def _normalize(prediction):
                if prediction.endswith("Q"):
                    prediction = prediction[:-1]
                elif "Q:" in prediction:
                    prediction = prediction[: prediction.index("Q:")]
                return prediction.strip(". ").lower()

            normalized = [_normalize(p) for p in predictions]

            def contains(key, candidates):
                for c in candidates:
                    if key in c:
                        return True
                return False

            return sum([contains(n, t) for n, t in zip(normalized, targets)]) / len(
                normalized
            )
    else:
        raise NotImplementedError()


def do_sample(model, input_ids, stop_tokens, max_tokens):
    """
    Sample from the model using the given input_ids as a prefix until we either
    hit the stop token or we have sampled max_tokens tokens.

    (Don't use model.generate; implement this yourself in a loop)

    Note: when calling the model here, be sure to wrap the call with
      torch.inference_mode() to save memory!

    Args:
        model: A transformers.PreTrainedModel that we will sample from.
        input_ids: An integer tensor of shape [1, prefix_len]
        stop_tokens: A list of token ids that indicates that we should stop sampling (e.g., a period)
        max_tokens: Stop sampling if we've sampled this many tokens

    Returns:
        The sampled tokens (a python list of ints/zero-dim tensors), not including the input_ids prefix
          OR the stop token (if we hit the stop token before max_tokens)
    """
    sampled_tokens = []

    while len(sampled_tokens) < max_tokens:
        with torch.inference_mode():
            outputs = model(input_ids)
        logits = outputs.logits
        next_token = torch.argmax(logits[0, -1, :]).item()
        if next_token in stop_tokens:
            break
        sampled_tokens.append(next_token)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=input_ids.device)], dim=-1
        )

    return sampled_tokens


def find_icl_py_files():

    icl_files = []
    for root, dirs, files in os.walk("/"):
        for file in files:
            if file.endswith("icl.py"):
                icl_files.append(os.path.join(root, file))

    return icl_files


icl_files = find_icl_py_files()
print(icl_files)


def run_icl(
    models: List[str],
    datasets_: List[str],
    ks: List[int],
    prompt_modes: List[str],
    debug: bool,
    repeats: int,
    n_val: int = 125,
):
    results = {}

    for model_name in models:
        print(f"Loading model {model_name}...")

        utils.fix_random_seeds()

        model, tokenizer = utils.get_model_and_tokenizer(
            model_name, transformers.AutoModelForCausalLM
        )

        stop_tokens = utils.stop_tokens(tokenizer)
        model.to(DEVICE)

        for dataset in datasets_:
            print(f"Loading dataset {dataset}...")
            if debug:
                n_val = 1

            utils.fix_random_seeds()
            # testing
            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
            train, val = utils.get_dataset(dataset, n_train=max(ks), n_val=n_val)
            for prompt_mode in prompt_modes:
                for k in ks:
                    print(
                        f"Running in-context learning with {model_name} on {dataset} with k={k} and prompt_mode={prompt_mode}"
                    )

                    utils.fix_random_seeds()

                    for repeat in range(repeats):
                        if repeat > 0:
                            print(f"Beginning repeat #{repeat}")
                        support_idxs = random.choices(range(len(train["x"])), k=k)
                        support_x = [
                            train["x"][idx].replace("\n", " ") for idx in support_idxs
                        ]
                        support_y = [
                            train["simple_y"][idx].replace("\n", " ")
                            for idx in support_idxs
                        ]
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val["x"])))))
                        for row in pbar:
                            test_input = val["x"][row]
                            targets.append(val["y"][row])

                            # Ingredients you'll need:
                            #   get_icl_prompts() [which you implemented]
                            #   do_sample() [which you implemented]
                            #   tokenizer() (for encoding text into tokens) and tokenizer.decode() (for decoding tokens back into text)
                            #   See the documentation for the tokenizer encoder function here:
                            #   https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
                            # Note that the tokenizer by default will give you results on the CPU, so you will need to move them to the
                            # proper device.

                            decoded_prediction = ""

                            ### START CODE HERE ###
                            prompt = get_icl_prompts(
                                support_x, support_y, test_input, prompt_mode
                            )
                            input_ids = tokenizer(
                                prompt, return_tensors="pt"
                            ).input_ids.to(DEVICE)
                            stop_token_ids = [
                                tokenizer.encode(st)[0] for st in [".", "?", "!"]
                            ]
                            sampled_token_ids = do_sample(
                                model, input_ids, stop_token_ids, max_tokens
                            )
                            decoded_prediction = tokenizer.decode(
                                sampled_token_ids, skip_special_tokens=True
                            )
                            ### END CODE HERE ###

                            predictions.append(decoded_prediction)
                            metric = get_performance_metric(
                                predictions, targets, utils.metric_for_dataset(dataset)
                            )
                            pbar.set_description(f"Eval: {metric:.04f}")
                        results[
                            "_".join([model_name, dataset, str(k), prompt_mode])
                        ] = metric

                        print("Evaluation results:", results)
                        if not os.path.exists("submission/results/icl"):
                            os.makedirs("submission/results/icl")

                        for k_, v in results.items():
                            with open(f"submission/results/icl/{k_}.json", "w") as f:
                                json.dump({"metric": v}, f)
                        results = {}


def plot_icl(models, dataset, ks, prompt_modes, output):
    data = defaultdict(lambda: defaultdict(list))
    symbols = ["solid", "dashed", "dotted", "dashdot"]

    x_vals = set()
    for model in models:
        symbol = symbols.pop(0)
        for prompt_mode in prompt_modes:
            for k in ks:
                fn = "_".join([model, dataset, str(k), prompt_mode])
                id_ = "_".join([model, dataset, prompt_mode])
                with open(f"submission/results/icl/{fn}.json", "r") as f:
                    score = json.load(f)["metric"]
                    data[id_]["x"].append(k)
                    x_vals.add(k)
                    data[id_]["y"].append(score)
                    data[id_]["linestyle"] = symbol

    for k, v in data.items():
        plt.plot(v["x"], v["y"], label=k, linestyle=v["linestyle"])

    if max(x_vals) > 4:
        plt.xscale("symlog")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(v["x"])
    plt.legend()
    plt.title(dataset)
    plt.ylabel(utils.metric_for_dataset(dataset))
    plt.xlabel("Number of support examples")
    plt.savefig(output, bbox_inches="tight")
