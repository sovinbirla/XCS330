#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections, os, pickle, gzip, shutil
import torch
import numpy as np
import itertools
import json

import transformers

import utils

# os.environ["DEVICE"] = "gpu"
DEVICE = torch.device("cpu")

from graderUtil import graded, CourseTestRunner, GradedTestCase, blockPrint, enablePrint

blockPrint()

# Import submission
import submission

enablePrint()


#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################


#########
# TESTS #
#########


class Test_1a(GradedTestCase):

    def setUp(self):

        # Cache the datasets and models needed to avoid timeout on the test cases
        utils.get_model_and_tokenizer("bert-tiny", transformers.AutoModelForCausalLM)
        utils.get_model_and_tokenizer(
            "bert-tiny", transformers.AutoModelForSequenceClassification, num_labels=5
        )
        utils.get_dataset(dataset="amazon", n_train=1, n_val=125)

    @graded()
    def test_0(self):
        """1a-0-basic: Basic test case for testing the number of parameters to fine tune in mode 'all'."""

        model, _ = utils.get_model_and_tokenizer(
            "bert-tiny", transformers.AutoModelForCausalLM
        )

        parameters = [
            parameter for parameter in submission.parameters_to_fine_tune(model, "all")
        ]

        # Check that the number of parameters to be optimized match
        self.assertTrue(
            len(parameters) == 42,
            "Incorrect number of parameters to be optimized returned by parameters_to_fine_tune ! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=10, is_hidden=True)
    def test_1(self):
        """1a-1-hidden: Hidden test case for testing the loss and accuracy for 2D inputs."""

        utils.fix_random_seeds()

        model, tokenizer = utils.get_model_and_tokenizer(
            "bert-tiny", transformers.AutoModelForSequenceClassification, num_labels=5
        )

        train, _ = utils.get_dataset(dataset="amazon", n_train=1, n_val=125)
        x, y = train["x"][:5], train["y"][:5]
        batch = np.random.randint(0, len(x), 8)
        x_ = tokenizer(
            [x[i] for i in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
        ).to(DEVICE)
        y_ = torch.tensor([y[i] for i in batch], device=DEVICE)

        logits = model(**x_).logits
        loss = submission.get_loss(logits, y_)
        acc = submission.get_acc(logits, y_)

        ### BEGIN_HIDE ###
        ### END_HIDE ###


class Test_1b(GradedTestCase):

    @graded(timeout=-1, is_hidden=True)
    def test_0(self):
        """1b-0-hidden: Hidden test case for testing the generated files for Q1."""

        question = "ft"
        models = ["bert-tiny", "bert-med"]
        datasets = ["amazon"]
        ks = [1, 8, 128]
        modes = ["all"]

        ### BEGIN_HIDE ###
        ### END_HIDE ###

        for model, dataset, k, mode in itertools.product(models, datasets, ks, modes):
            output_file = f"{model}_{dataset}_{k}_{mode}"

            # Check that the file exists
            self.assertTrue(
                os.path.exists(f"submission/results/{question}/{output_file}.json"),
                f"Output file submission/results/{question}/{output_file}.json does not exist! Please follow all requirements outlined in the function comments and the writeup.",
            )

            ### BEGIN_HIDE ###
            ### END_HIDE ###


class Test_2a(GradedTestCase):

    def setUp(self):

        # Cache the datasets and models needed to avoid timeout on the test cases
        utils.get_model_and_tokenizer("med", transformers.AutoModelForCausalLM)

        self.get_icl_prompts = submission.get_icl_prompts
        self.sol_get_icl_prompts = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.get_icl_prompts
        )

        self.do_sample = submission.do_sample
        self.sol_do_sample = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.do_sample
        )

        self.prompt_ids = torch.tensor(
            [
                [
                    464,
                    4141,
                    12,
                    11990,
                    609,
                    38630,
                    16055,
                    12052,
                    11,
                    636,
                    286,
                    350,
                    1142,
                    375,
                    15868,
                    446,
                    11,
                    3382,
                    284,
                    1382,
                    319,
                    262,
                    2524,
                    286,
                    262,
                    11773,
                    1233,
                    14920,
                    1474,
                    1879,
                    1313,
                    11,
                    543,
                    468,
                    407,
                    587,
                    973,
                    329,
                    1478,
                    812,
                    13,
                    383,
                    2656,
                    1233,
                    14920,
                    6832,
                    423,
                    1541,
                    587,
                    33359,
                    13,
                    21913,
                    290,
                    11344,
                    2594,
                    5583,
                    8900,
                    15796,
                    9847,
                    531,
                    262,
                    649,
                    1233,
                    14920,
                    561,
                    307,
                    7062,
                    13,
                    632,
                    561,
                    1612,
                    281,
                    2620,
                    286,
                    838,
                    4,
                    287,
                    262,
                    1664,
                    338,
                    26868,
                    39501,
                    1233,
                    4509,
                    5339,
                    13,
                    609,
                    38630,
                    318,
                    262,
                    1218,
                    4094,
                    1664,
                    287,
                    46755,
                    39501,
                    11,
                    351,
                    546,
                    1160,
                    4,
                    286,
                    262,
                    1910,
                    13,
                    30305,
                    329,
                    257,
                    1688,
                    649,
                    1233,
                    14920,
                    287,
                    2531,
                    893,
                    485,
                    423,
                    587,
                    6325,
                    416,
                    3461,
                    323,
                    43321,
                    13,
                    383,
                    35090,
                    19862,
                    10571,
                    357,
                    23055,
                    19862,
                    4608,
                    8,
                    3952,
                    318,
                    1363,
                    284,
                    262,
                    4387,
                    10368,
                    286,
                    262,
                    995,
                    338,
                    5637,
                    530,
                    12,
                    25311,
                    276,
                    9529,
                    259,
                    420,
                    27498,
                    13,
                    198,
                    21300,
                    942,
                    3751,
                    3952,
                    3085,
                    1866,
                    17247,
                    290,
                    788,
                    32331,
                    262,
                    31134,
                    832,
                    262,
                    3952,
                    13,
                    198,
                    1026,
                    318,
                    407,
                    1900,
                    810,
                    262,
                    31134,
                    338,
                    2802,
                    318,
                    11,
                    475,
                    49879,
                    318,
                    257,
                    2219,
                    1917,
                    287,
                    16385,
                    1723,
                    64,
                    13,
                    198,
                    14565,
                    2056,
                    8636,
                    326,
                    1105,
                    9529,
                    259,
                    420,
                    27498,
                    547,
                    2923,
                    416,
                    745,
                    17892,
                    1201,
                    3269,
                    428,
                    614,
                    13,
                ]
            ]
        )

    @graded(timeout=1)
    def test_0(self):
        """2a-0-basic: Basic test case for checking your own prompt format is different from the ones we have shown."""

        support_inputs = [
            "Sandra travelled to the kitchen. John went to the office. Where is Sandra?"
        ]
        support_labels = ["kitchen"]
        test_input = "Sandra went to the office. Daniel went back to the garden. Where is Sandra?"

        # Get all types of prompts
        utils.fix_random_seeds()
        prompt_qa = self.get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="qa"
        )

        utils.fix_random_seeds()
        prompt_none = self.get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="none"
        )

        utils.fix_random_seeds()
        prompt_tldr = self.get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="tldr"
        )

        utils.fix_random_seeds()
        prompt_custom = self.get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="custom"
        )

        # Check that custom prompt is different than the rest of our specified formats
        self.assertTrue(
            prompt_custom != prompt_qa
            and prompt_custom != prompt_none
            and prompt_custom != prompt_tldr,
            "Custom prompt format from get_icl_prompts needs to be different than our given formats! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=1, is_hidden=True)
    def test_1(self):
        """2a-1-hidden: Hidden test case for checking the generated prompts."""

        support_inputs = [
            "Daniel moved to the bathroom. Mary went back to the hallway. Where is Mary?",
            "Daniel moved to the bathroom. John travelled to the garden. Where is John?",
            "Sandra journeyed to the office. Mary moved to the bedroom. Where is Mary?",
        ]
        support_labels = ["hallway", "garden", "bedroom"]
        test_input = "Sandra journeyed to the bedroom. Daniel moved to the kitchen. Where is Daniel?"

        # Get student prompts
        utils.fix_random_seeds()
        prompt_qa = self.get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="qa"
        )

        utils.fix_random_seeds()
        prompt_none = self.get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="none"
        )

        utils.fix_random_seeds()
        prompt_tldr = self.get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="tldr"
        )

        # Get solution prompts
        utils.fix_random_seeds()
        sol_prompt_qa = self.sol_get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="qa"
        )

        utils.fix_random_seeds()
        sol_prompt_none = self.sol_get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="none"
        )

        utils.fix_random_seeds()
        sol_prompt_tldr = self.sol_get_icl_prompts(
            support_inputs, support_labels, test_input, prompt_mode="tldr"
        )

        # Compare the prompts
        self.assertTrue(
            prompt_qa == sol_prompt_qa,
            "`qa` prompt not matching! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            prompt_none == sol_prompt_none,
            "`none` prompt not matching! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            prompt_tldr == sol_prompt_tldr,
            "`tldr` prompt not matching! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=86400)
    def test_2(self):
        """2a-2-basic: Basic test case for checking that do_sample does not include the input_ids prefix OR the stop token."""

        model, tokenizer = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )
        stop_tokens = utils.stop_tokens(tokenizer)

        max_tokens = utils.max_sampled_tokens_for_dataset("xsum")

        sampled_tokens = self.do_sample(
            model=model.to(DEVICE),
            input_ids=self.prompt_ids.to(DEVICE),
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
        )

        # Check that the sampled tokens do not include the input_ids prefix
        self.assertFalse(
            all(x in sampled_tokens for x in list(self.prompt_ids.squeeze().numpy())),
            "do_sample should not include the input_idx prefix! Please follow all requirements outlined in the function comments and the writeup.",
        )

        # Check that the sampled tokens do not include the stop tokens
        self.assertFalse(
            any(x in sampled_tokens for x in stop_tokens),
            "do_sample should not include any stop token! Please follow all requirements outlined in the function comments and the writeup.",
        )

        # Check that we sampled at most max_tokens tokens
        self.assertTrue(
            len(sampled_tokens) > 0 and len(sampled_tokens) <= max_tokens,
            f"do_sample should at most sample {max_tokens} tokens! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=60, is_hidden=True)
    def test_3(self):
        """2a-3-hidden: Hidden test case for checking that do_sample generate tokens match."""

        model, tokenizer = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )
        stop_tokens = utils.stop_tokens(tokenizer)

        max_tokens = utils.max_sampled_tokens_for_dataset("xsum")

        utils.fix_random_seeds()

        sampled_tokens = self.do_sample(
            model=model.to(DEVICE),
            input_ids=self.prompt_ids.to(DEVICE),
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
        )

        utils.fix_random_seeds()

        sol_sampled_tokens = self.sol_do_sample(
            model=model.to(DEVICE),
            input_ids=self.prompt_ids.to(DEVICE),
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
        )

        # Check that sampled tokens match
        self.assertTrue(
            sampled_tokens == sol_sampled_tokens,
            f"Issue in do_sample! Please follow all requirements outlined in the function comments and the writeup.",
        )


class Test_2b(GradedTestCase):

    @graded(timeout=-1, is_hidden=True)
    def test_0(self):
        """2b-0-hidden: Hidden test case for testing the generated files for Q2b."""

        question = "icl"
        models = ["med", "full"]
        datasets = ["babi"]
        ks = [0, 1, 16]
        prompt_modes = ["qa"]

        ### BEGIN_HIDE ###
        ### END_HIDE ###

        for model, dataset, k, prompt_mode in itertools.product(
            models, datasets, ks, prompt_modes
        ):
            output_file = f"{model}_{dataset}_{k}_{prompt_mode}"

            # Check that the file exists
            self.assertTrue(
                os.path.exists(f"submission/results/{question}/{output_file}.json"),
                f"Output file submission/results/{question}/{output_file}.json does not exist! Please follow all requirements outlined in the function comments and the writeup.",
            )

            ### BEGIN_HIDE ###
            ### END_HIDE ###


class Test_2c(GradedTestCase):

    @graded(timeout=-1, is_hidden=True)
    def test_0(self):
        """2c-0-hidden: Hidden test case for testing the generated files for Q2c."""

        question = "icl"
        models = ["med", "full"]
        datasets = ["xsum"]
        ks = [0, 1, 4]
        prompt_modes = ["none", "tldr"]

        ### BEGIN_HIDE ###
        ### END_HIDE ###

        for model, dataset, k, prompt_mode in itertools.product(
            models, datasets, ks, prompt_modes
        ):
            output_file = f"{model}_{dataset}_{k}_{prompt_mode}"

            # Check that the file exists
            self.assertTrue(
                os.path.exists(f"submission/results/{question}/{output_file}.json"),
                f"Output file submission/results/{question}/{output_file}.json does not exist! Please follow all requirements outlined in the function comments and the writeup.",
            )

            ### BEGIN_HIDE ###
            ### END_HIDE ###


class Test_3a(GradedTestCase):

    def setUp(self):

        # Cache the datasets and models needed to avoid timeout on the test cases
        utils.get_model_and_tokenizer("med", transformers.AutoModelForCausalLM)

        self.parameters_to_fine_tune = submission.parameters_to_fine_tune
        self.sol_parameters_to_fine_tune = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.parameters_to_fine_tune
        )

    @graded(timeout=10)
    def test_0(self):
        """3a-0-basic: Basic test case for testing the parameters to fine tune in mode 'last', 'first' and 'middle'."""

        model, _ = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )

        last_parameters = [
            parameter for parameter in self.parameters_to_fine_tune(model, "last")
        ]
        first_parameters = [
            parameter for parameter in self.parameters_to_fine_tune(model, "first")
        ]
        middle_parameters = [
            parameter for parameter in self.parameters_to_fine_tune(model, "middle")
        ]

        # Check that the number of parameters to be optimized match
        self.assertTrue(
            len(last_parameters) == 24,
            "Incorrect number of parameters to be optimized returned by parameters_to_fine_tune for `last` mode! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            len(first_parameters) == 24,
            "Incorrect number of parameters to be optimized returned by parameters_to_fine_tune for `first` mode! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            len(middle_parameters) == 24,
            "Incorrect number of parameters to be optimized returned by parameters_to_fine_tune for `middle` mode! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=10, is_hidden=True)
    def test_1(self):
        """3a-1-hidden: Hidden test case for testing the parameters to fine tune in mode 'last', 'first' and 'middle' are the ones expected."""

        model, _ = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )

        last_parameters = [
            parameter for parameter in self.parameters_to_fine_tune(model, "last")
        ]
        first_parameters = [
            parameter for parameter in self.parameters_to_fine_tune(model, "first")
        ]

        utils.fix_random_seeds()
        middle_parameters = [
            parameter for parameter in self.parameters_to_fine_tune(model, "middle")
        ]

        sol_last_parameters = [
            parameter for parameter in self.sol_parameters_to_fine_tune(model, "last")
        ]
        sol_first_parameters = [
            parameter for parameter in self.sol_parameters_to_fine_tune(model, "first")
        ]

        utils.fix_random_seeds()
        sol_middle_parameters = [
            parameter for parameter in self.sol_parameters_to_fine_tune(model, "middle")
        ]

        # Check that the number of parameters to be optimized match
        self.assertTrue(
            all(
                any([torch.equal(x, y) for y in last_parameters])
                for x in sol_last_parameters
            ),
            "Parameters to be optimized returned by parameters_to_fine_tune for `last` mode do not match our solution! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            all(
                any([torch.equal(x, y) for y in first_parameters])
                for x in sol_first_parameters
            ),
            "Parameters to be optimized returned by parameters_to_fine_tune for `first` mode do not match our solution! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            all(
                any([torch.equal(x, y) for y in middle_parameters])
                for x in sol_middle_parameters
            ),
            "Parameters to be optimized returned by parameters_to_fine_tune for `middle` mode do not match our solution! Please follow all requirements outlined in the function comments and the writeup.",
        )


class Test_3c(GradedTestCase):

    def setUp(self):

        # Cache the datasets and models needed to avoid timeout on the test cases
        utils.get_model_and_tokenizer("med", transformers.AutoModelForCausalLM)
        utils.get_dataset(dataset="xsum", n_train=8, n_val=125)

        self.LoRAConv1DWrapper = submission.LoRAConv1DWrapper
        self.sol_LoRAConv1DWrapper = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.LoRAConv1DWrapper
        )

        self.parameters_to_fine_tune = submission.parameters_to_fine_tune
        self.sol_parameters_to_fine_tune = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.parameters_to_fine_tune
        )

        self.tokenize_gpt2_batch = submission.tokenize_gpt2_batch
        self.sol_tokenize_gpt2_batch = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.tokenize_gpt2_batch
        )

        self.get_acc = submission.get_acc
        self.sol_get_acc = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.get_acc
        )

        self.get_loss = submission.get_loss
        self.sol_get_loss = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.get_loss
        )

    @graded(timeout=10)
    def test_0(self):
        """3c-0-basic: Basic test case for checking the internal parameters of LoRAConv1DWrapper."""

        model, _ = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )
        lora_rank = 4

        lora_layer = self.LoRAConv1DWrapper(model.transformer.h[0].mlp.c_fc, lora_rank)

        d1, d2 = lora_layer.base_module.weight.shape

        # print(d1, d2)
        # print(lora_layer.lora_A.shape, lora_layer.lora_B.shape)

        # Check that the lora parameters have the right dimensions
        self.assertTrue(
            lora_layer.lora_A.shape == (d1, lora_rank),
            "Incorrect shape of the `A` matrix of the LoRA layer! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            lora_layer.lora_B.shape == (d2, lora_rank),
            "Incorrect shape of the `B` matrix of the LoRA layer! Please follow all requirements outlined in the function comments and the writeup.",
        )

        # Check that the lora parameters were not initialized to zeros
        self.assertFalse(
            torch.equal(lora_layer.lora_A, torch.zeros((d1, lora_rank)))
            and torch.equal(lora_layer.lora_B, torch.zeros((d2, lora_rank))),
            "`A` and `B` matrices of the LoRA layer shouldn't not be initialized to 0! Please follow all requirements outlined in the function comments and the writeup.",
        )

        # Check whether the pretrained model weights were frozen as described in the paper
        self.assertFalse(
            lora_layer.base_module.weight.requires_grad,
            "As per the LoRA paper, the pretrained model weights should be frozen! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=10, is_hidden=True)
    def test_1(self):
        """3c-1-hidden: Hidden test case for further checks on the internal parameters of LoRAConv1DWrapper."""

        model, _ = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )
        lora_rank = 4

        lora_layer = self.LoRAConv1DWrapper(model.transformer.h[0].mlp.c_fc, lora_rank)

        d1, d2 = lora_layer.base_module.weight.shape

        # print(d1, d2)

        ### BEGIN_HIDE ###
        ### END_HIDE ###

    @graded(timeout=10, is_hidden=True)
    def test_2(self):
        """3c-2-hidden: Hidden test case for checking the forward method of LoRAConv1DWrapper."""

        utils.fix_random_seeds()

        model, _ = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )
        lora_rank = 4

        x = torch.randn((1, 111, 1024))

        lora_layer = self.LoRAConv1DWrapper(model.transformer.h[0].mlp.c_fc, lora_rank)

        out = lora_layer(x)

        sol_lora_layer = self.sol_LoRAConv1DWrapper(
            model.transformer.h[0].mlp.c_fc, lora_rank
        )
        sol_out = sol_lora_layer(x)

        print(out)
        print(sol_out)
        # Check that the outputs match
        self.assertTrue(
            torch.equal(sol_out, out),
            "LoRAConv1DWrapper outputs do not match! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=10)
    def test_3(self):
        """3c-3-basic:  Basic test case for testing the parameters to fine tune in mode 'loraN'."""

        model, _ = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )
        mode = "lora4"

        for m in model.transformer.h:
            m.mlp.c_fc = self.LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = self.LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))
            m.attn.c_attn = self.LoRAConv1DWrapper(m.attn.c_attn, int(mode[4:]))

        loraN_parameters = [
            parameter for parameter in self.parameters_to_fine_tune(model, mode)
        ]

        # Check that the number of parameters to be optimized match
        self.assertTrue(
            len(loraN_parameters) == 144,
            "Incorrect number of parameters to be optimized returned by parameters_to_fine_tune for `lora` mode! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=10)
    def test_4(self):
        """3c-4-basic: Basic test case for testing the tokenize_gpt2_batch outputs dimensions."""

        utils.fix_random_seeds()

        _, tokenizer = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )

        train, _ = utils.get_dataset(dataset="xsum", n_train=8, n_val=125)
        x, y = train["x"][:8], train["simple_y"][:8]
        all_both = self.tokenize_gpt2_batch(tokenizer, x, y)

        # print(
        #     all_both["input_ids"].shape,
        #     all_both["attention_mask"].shape,
        #     all_both["labels"].shape,
        # )

        # Check the output elements are all of shape (batch_size, sequence_length)
        self.assertTrue(
            all_both["input_ids"].shape == (8, 124),
            "Incorrect shape of the input_ids element of tokenize_gpt2_batch output! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            all_both["attention_mask"].shape == (8, 124),
            "Incorrect shape of the attention_mask element of tokenize_gpt2_batch output! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            all_both["labels"].shape == (8, 124),
            "Incorrect shape of the labels element of tokenize_gpt2_batch output! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=15, is_hidden=True)
    def test_5(self):
        """3c-5-hidden: Hidden test case for testing the tokenize_gpt2_batch outputs."""

        utils.fix_random_seeds()

        _, tokenizer = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )

        train, _ = utils.get_dataset(dataset="xsum", n_train=8, n_val=125)
        x, y = train["x"][:8], train["simple_y"][:8]

        all_both = self.tokenize_gpt2_batch(tokenizer, x, y)
        sol_all_both = self.sol_tokenize_gpt2_batch(tokenizer, x, y)

        # Check the output elements are all matching
        self.assertTrue(
            torch.equal(all_both["input_ids"], sol_all_both["input_ids"]),
            "Incorrect input_ids element of tokenize_gpt2_batch output! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            torch.equal(all_both["attention_mask"], sol_all_both["attention_mask"]),
            "Incorrect attention_mask element of tokenize_gpt2_batch output! Please follow all requirements outlined in the function comments and the writeup.",
        )
        self.assertTrue(
            torch.equal(all_both["labels"], sol_all_both["labels"]),
            "Incorrect labels element of tokenize_gpt2_batch output! Please follow all requirements outlined in the function comments and the writeup.",
        )

    @graded(timeout=20, is_hidden=True)
    def test_6(self):
        """3c-6-hidden: Hidden test case for testing the loss and accuracy for 3D inputs."""

        utils.fix_random_seeds()

        model, tokenizer = utils.get_model_and_tokenizer(
            "med", transformers.AutoModelForCausalLM
        )

        train, _ = utils.get_dataset(dataset="xsum", n_train=8, n_val=125)
        x, y = train["x"][:8], train["simple_y"][:8]
        all_both = self.sol_tokenize_gpt2_batch(tokenizer, x, y)

        accs, losses = [], []
        for idx in range(len(list(all_both.values())[0])):
            d = {k: v[idx : idx + 1] for k, v in all_both.items()}

            logits = model(**d).logits

            acc = self.get_acc(logits, d["labels"])
            accs.append(acc)

            loss = self.get_loss(logits, d["labels"])
            losses.append(loss)
        total_acc = sum(accs) / len(accs)
        total_loss = sum(losses)

        ### BEGIN_HIDE ###
        ### END_HIDE ###


class Test_3d(GradedTestCase):

    @graded(timeout=-1, is_hidden=True)
    def test_0(self):
        """3d-0-hidden: Hidden test case for testing the generated files for Q3."""

        question = "ft"
        models = ["med"]
        datasets = ["xsum", "babi"]
        ks = [0, 1, 8, 128]
        modes = ["first", "last", "middle", "lora4", "lora16"]

        ### BEGIN_HIDE ###
        ### END_HIDE ###

        for model, dataset, k, mode in itertools.product(models, datasets, ks, modes):
            output_file = f"{model}_{dataset}_{k}_{mode}"

            # Check that the file exists
            self.assertTrue(
                os.path.exists(f"submission/results/{question}/{output_file}.json"),
                f"Output file submission/results/{question}/{output_file}.json does not exist! Please follow all requirements outlined in the function comments and the writeup.",
            )

            ### BEGIN_HIDE ###
            ### END_HIDE ###


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":

    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
