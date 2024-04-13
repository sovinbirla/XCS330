from typing import List, Tuple
import argparse
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
import utils
import copy
import numpy as np
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import itertools
from .icl import get_icl_prompts, do_sample, get_performance_metric
import tqdm
import random

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

print("Fine-tuning using device: ", DEVICE)


class LoRAConv1DWrapper(nn.Module):
    def __init__(self, conv1dmodule: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = conv1dmodule

        ###
        ### Set up your LoRA-augmented layer here.
        ### You should initialize your parameters so that the residual matrix AB^T is zero,
        ###     but be careful how you do this (i.e., make sure you eventually get
        ###     non-zero gradients to both matrices during fine-tuning)!
        ### Initialization hint: what do the gradients look like after 1 and 2 steps of fine-tuning
        ###     if you initialize both A and B to zero? What about if just one is zero?
        ###

        # self.lora_A, self.lora_B = None, None
        self.base_module.weight.requires_grad = False
        weight_shape = self.base_module.weight.shape
        self.lora_A = nn.Parameter(torch.randn(weight_shape[0], lora_rank))
        self.lora_B = nn.Parameter(torch.randn(weight_shape[1], lora_rank))

    def forward(self, x):
        ###
        ### Perform the forward pass of your LoRA-augmented layer here.
        ### Note: you don't need to ever explicitly construct the matrix AB^T.
        ### Hint: matrix multiplication is associative.
        ###
        #############################
        ### START CODE HERE ###
        lora_augmentation = F.linear(x, torch.matmul(self.lora_A, self.lora_B.T).T)
        return self.base_module(x) + lora_augmentation


### END CODE HERE ###


def parameters_to_fine_tune(model: nn.Module, mode: str) -> List:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)

    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """

    if mode == "all":
        return model.parameters()
    elif mode == "last":
        return list(model.transformer.h[-2:].parameters())
    elif mode == "first":
        return list(model.transformer.h[:2].parameters())
    elif mode == "middle":
        num_layers = len(model.transformer.h)
        assert num_layers % 2 == 0  # Ensure even number of layers
        mid = num_layers // 2
        return list(model.transformer.h[mid - 1 : mid + 1].parameters())
    elif mode.startswith("lora"):
        lora_params = []
        for name, module in model.named_modules():
            if isinstance(module, LoRAConv1DWrapper):
                lora_params.extend([module.lora_A, module.lora_B])
        return lora_params
    else:
        raise NotImplementedError()


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the cross-entropy loss for either sequence classification or generation.

    For generation, you'll need to deal with the fact that different sequences witihn
      the batch are different lengths, and the targets tensor includes some mask
      values (-100). The average loss is the *average loss over all non-masked timesteps*.
      You'll also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor
        of *UNNORMALIZED* logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.

    Returns:
      A zero-dim tensor representing the average cross-entropy loss over all batch
        elements (and sequence timesteps, if applicable)
    """

    loss = None
    if logits.dim() == 2:  # Classification
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
    elif logits.dim() == 3:  # Generation
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        active_loss = targets.view(-1) != -100
        active_logits = logits.view(-1, logits.size(-1))[active_loss]
        active_labels = targets.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels)
    else:
        raise ValueError(
            f"Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}"
        )

    return loss


def get_acc(logits, targets):
    """
    Computes the exact match accuracy for either sequence classification or generation. i.e.,
      the fraction of predictions for which the most likely class/token equals the target.

    For generation, you'll need to deal with the fact that different sequences within
      the batch are different lengths, and the targets tensor includes some mask
      values (-100). The average accuracy is the *average accuracy over all non-masked timesteps*.
      You'll also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor of logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.

    Returns:
      A *scalar* representing the average exact-match accuracy over all non-masked batch
        elements (and sequence timesteps, if applicable)
    """

    if logits.dim() == 2:  # Classification
        preds = torch.argmax(logits, dim=1)
        correct = preds == targets
        acc = correct.float().mean()
    elif logits.dim() == 3:  # Generation
        preds = torch.argmax(logits, dim=2)
        mask = targets != -100
        correct = (preds == targets) & mask
        acc = correct.float().sum() / mask.float().sum()
    else:
        raise ValueError(
            f"Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}"
        )
    return acc


def ft_bert(model, tok, x, y, mode, debug, batch_size=8):
    model = copy.deepcopy(model)

    if mode.startswith("lora"):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=1e-4)
    all_x = tok(
        x, return_tensors="pt", padding=True, truncation=True, max_length=100
    ).to(DEVICE)
    all_y = torch.tensor(y, device=DEVICE)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = np.random.randint(0, len(x), batch_size)
        x_ = tok(
            [x[i] for i in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
        ).to(DEVICE)
        y_ = torch.tensor([y[i] for i in batch], device=DEVICE)
        logits = model(**x_).logits
        loss = get_loss(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if debug:
            break

        if step % 10 == 0:
            with torch.inference_mode():
                total_acc = get_acc(model(**all_x).logits, all_y)
            pbar.set_description(f"Fine-tuning acc: {total_acc:.04f}")
            if total_acc > 0.75:
                break
    return model


def tokenize_gpt2_batch(tokenizer, x, y):
    """
    Implement the tokenization step for a batch of examples for GPT-2.

    Args:
        tokenizer: a GPT2Tokenizer that you can call and receive a dictionary of:
          - input_ids: a list (or tensor) of token ids
          - attention_mask: a list (or tensor) of 1s and 0s indicating which tokens
              are padding (if you requested padding and tensors from the tokenizer)
        x: a list of strings, each of which is the input for a single example
        y: a list of strings, each of which is a *target* for a single example

    Returns:
        A dictionary with the following keys:
            - input_ids: a tensor of shape [batch_size, sequence_length]
                containing the token ids
            - attention_mask: a tensor of shape [batch_size, sequence_length]
                containing 1s and 0s indicating which tokens are padding
            - labels: a tensor of shape [batch_size, sequence_length] containing
                the target token ids, with -100 for non-target tokens (i.e., the
                tokens in the input part of each example or padding tokens)
        where sequence_length is determined by the (x, y) pair whose tokenized
        length is the longest in the batch. The other sequences should be padded to
        this length (you can get the tokenizer to handle this padding!).

    Example:
        >>> x = ['Who is the singer for the band Queen?', 'What is the capital of France?']
        >>> y = ['Freddie Mercury', 'Paris']
        >>> tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        >>> tokenizer_dict = tokenizer([x_ + y_ for x_, y_ in zip(x, y)], return_tensors='pt', padding=True)
        >>> tokenizer_dict['input_ids']
        tensor([[ 8241,   318,   262, 14015,   329,   262,  4097,  7542,    30, 30847, 11979, 21673],
                [ 2061,   318,   262,  3139,   286,  4881,    30, 40313, 50256, 50256, 50256, 50256]])
        >>> tokenizer_dict['attention_mask']
        tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
        >>> tokenizer(x)['input_ids']
        [[8241, 318, 262, 14015, 329, 262, 4097, 7542, 30],
         [2061, 318, 262, 3139, 286, 4881, 30]]
        >>> tokenizer(y)['input_ids']
        [[30847, 11979, 21673],
         [40313]]

        In this case, our labels should look like:
        [[-100, -100, -100, -100, -100, -100, -100, -100,   -100,  30847, 11979, 21673],
         [-100, -100, -100, -100, -100, -100, -100,  40313, -100, -100,  -100,  -100]]
        Note we've replaced padding tokens and the input prefix for each example
            with -100, leaving only the tokens in y.

        Other note: you can add new keys (such as 'labels') to the dictionary
            returned by the tokenizer without creating a new dictionary.
    """
    tokenized_sequences = None

    # Tokenize x and y together
    tokenized_sequences = tokenizer(
        [x_ + " " + y_ for x_, y_ in zip(x, y)], return_tensors="pt", padding=True
    )

    # Create labels tensor
    labels = tokenized_sequences["input_ids"].clone()

    # Replace tokens from x and padding tokens with -100
    for i, (x_, y_) in enumerate(zip(x, y)):
        x_ids = tokenizer.encode(x_, add_special_tokens=False)
        labels[i, : len(x_ids)] = -100  # tokens from x
        labels[
            i, len(x_ids) + len(tokenizer.encode(y_, add_special_tokens=False)) :
        ] = -100  # padding tokens

    tokenized_sequences["labels"] = labels

    return tokenized_sequences


def add_prefixes(
    x: List[str], y: List[str], dataset: str
) -> Tuple[List[str], List[str]]:
    input_prefix = "" if utils.is_qa_dataset(dataset) else ""
    label_prefix = " In the" if utils.is_qa_dataset(dataset) else " TL;DR:"
    label_suffix = "." if utils.is_qa_dataset(dataset) else ""

    x = [input_prefix + x_.replace("\n", " ") + label_prefix for x_ in x]
    y = [" " + y_.replace("\n", " ") + label_suffix for y_ in y]

    return x, y


def ft_gpt2(model, tok, x, y, mode, dataset, batch_size=8, grad_accum=8):
    x, y = add_prefixes(x, y, dataset)

    model = copy.deepcopy(model)

    if mode.startswith("lora"):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))
            m.attn.c_attn = LoRAConv1DWrapper(m.attn.c_attn, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=2e-5)
    all_both = tokenize_gpt2_batch(tok, x, y)
    max_n = len(x) * 10
    pbar = tqdm.tqdm(range(max_n))
    idxs = []
    for step in pbar:
        model.train()

        if len(idxs) < batch_size // grad_accum:
            idxs = list(range(len(x)))
            random.shuffle(idxs)
        batch_idxs = idxs[: batch_size // grad_accum]
        idxs = idxs[batch_size // grad_accum :]

        # Outline:
        # 1. Sample a random minibatch of examples of size batch_size // grad_accum using the batch_idxs variable
        # 2. Tokenize the batch using the tokenize_gpt2_batch function you implemented
        # 3. Run the model on the batch, get the logits, and compute the loss using the get_loss function you implemented
        #      *NOTE 1* Pass `use_cache=False` when you call model() to avoid a huggingface warning
        #      *NOTE 2* You MUST compute the loss using your get_loss function applied to the model_output.logits.
        #        Don't use the loss attribute of the model output for training (you will not get credit for this).
        #        However, you can use the loss attribute of the model output to test your get_loss function (they should match).
        # 4. Backpropagate the loss (divided by the grad_accum parameter)
        # 5. Take a step of the optimizer and zero the model gradients ***only every grad_accum steps***
        #    Be careful that you don't take a step after the very first backward pass (i.e., when step == 0)
        # Note: the ** operator will unpack a dictionary into keyword arguments to a function (such as your model)
        #############################
        ### START CODE HERE ###
        batch_x = [x[i] for i in batch_idxs]
        batch_y = [y[i] for i in batch_idxs]
        batch_data = tokenize_gpt2_batch(tokenizer, batch_x, batch_y)
        logits = model(**batch_data, use_cache=False).logits
        loss = get_loss(logits, batch_data["labels"]) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        ### END CODE HERE ###

        if step % (grad_accum * 5) == 0:
            with torch.inference_mode():
                model.eval()
                accs = []
                for idx in range(len(list(all_both.values())[0])):
                    d = {k: v[idx : idx + 1] for k, v in all_both.items()}
                    acc = get_acc(model(**d).logits, d["labels"])
                    accs.append(acc)
                total_acc = sum(accs) / len(accs)
                pbar.set_description(f"Fine-tuning acc: {total_acc:.04f}")

            if total_acc >= utils.early_stop_thresold(dataset):
                print("Early stopping!")
                break
    return model


def eval(model, tok, val_data):
    x = tok(
        val_data["x"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
    ).to(DEVICE)
    y = torch.tensor(val_data["y"], device=DEVICE)
    with torch.inference_mode():
        logits = model(**x).logits
    return get_acc(logits, y)


def run_ft(
    models: List[str],
    datasets: List[str],
    ks: List[int],
    modes: List[str],
    debug: bool,
    repeats: int,
    n_val: int = 125,
):
    results = {}
    for dataset in datasets:

        utils.fix_random_seeds()

        if debug:
            n_val = 1
        train, val = utils.get_dataset(dataset, max(ks), n_val=n_val)
        for model_name, mode in itertools.product(models, modes):

            utils.fix_random_seeds()

            if dataset == "amazon":
                model, tokenizer = utils.get_model_and_tokenizer(
                    model_name,
                    transformers.AutoModelForSequenceClassification,
                    num_labels=5,
                )
            else:
                model, tokenizer = utils.get_model_and_tokenizer(
                    model_name, transformers.AutoModelForCausalLM
                )

            stop_tokens = utils.stop_tokens(tokenizer)

            for k in ks:
                print(
                    f"Fine-tuning {model_name} on {dataset} with k={k} and mode={mode}"
                )

                utils.fix_random_seeds()

                for repeat in range(repeats):
                    if repeat > 0:
                        print(f"Beginning repeat #{repeat}")
                    if dataset == "amazon":
                        fine_tuned = ft_bert(
                            model,
                            tokenizer,
                            train["x"][: k * 5],
                            train["y"][: k * 5],
                            mode,
                            debug,
                        )
                        val_acc = eval(fine_tuned, tokenizer, val)
                        results["_".join([model_name, dataset, str(k), mode])] = val_acc
                    else:
                        if k > 0:
                            fine_tuned = ft_gpt2(
                                model,
                                tokenizer,
                                train["x"][:k],
                                train["simple_y"][:k],
                                mode,
                                dataset,
                            )
                        else:
                            fine_tuned = copy.deepcopy(model)
                            fine_tuned.to(DEVICE)

                        fine_tuned.eval()
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val["x"])))))

                        for row in pbar:
                            test_input = val["x"][row]
                            targets.append(val["y"][row])
                            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
                            prompt_mode = (
                                "qa" if utils.is_qa_dataset(dataset) else "tldr"
                            )
                            prompt = get_icl_prompts(
                                [], [], test_input, prompt_mode=prompt_mode
                            )
                            input_ids = tokenizer(
                                prompt, return_tensors="pt"
                            ).input_ids.to(DEVICE)
                            sampled_tokens = do_sample(
                                fine_tuned, input_ids, stop_tokens, max_tokens
                            )
                            decoded = tokenizer.decode(sampled_tokens).strip()
                            predictions.append(decoded)
                            metric = get_performance_metric(
                                predictions, targets, utils.metric_for_dataset(dataset)
                            )
                            pbar.set_description(f"Eval: {metric:.04f}")
                        results["_".join([model_name, dataset, str(k), mode])] = metric

                    print(results)
                    question = "ft"
                    if not os.path.exists(f"submission/results/{question}"):
                        os.makedirs(f"submission/results/{question}")

                    for k_, v in results.items():
                        with open(f"submission/results/{question}/{k_}.json", "w") as f:
                            json.dump({"metric": v.tolist()}, f)
                    results = {}


def plot_ft(models, datasets, ks, modes, output):
    data = defaultdict(lambda: defaultdict(list))
    question = "ft"

    x_vals = set()
    for dataset in datasets:
        for model, mode in itertools.product(models, modes):
            for k in ks:
                fn = "_".join([model, dataset, str(k), mode])
                id_ = "_".join([model, dataset, mode])
                with open(f"submission/results/{question}/{fn}.json", "r") as f:
                    score = json.load(f)["metric"]
                    data[id_]["x"].append(k)
                    x_vals.add(k)
                    data[id_]["y"].append(score)

        for k, v in data.items():
            plt.plot(v["x"], v["y"], label=k)

    if max(x_vals) > 4:
        plt.xscale("symlog")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(sorted(x_vals))
    plt.legend()
    plt.title(" & ".join(datasets))
    plt.ylabel("/".join([utils.metric_for_dataset(dataset) for dataset in datasets]))
    plt.xlabel("Number of support examples")
    plt.savefig(output, bbox_inches="tight")
