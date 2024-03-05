#!/usr/bin/env python3
import unittest
import random
import argparse
import inspect
import os
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import torch
import torch.nn as nn
import omniglot
from google_drive_downloader import GoogleDriveDownloader as gdd

# Import submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

def fix_random_seeds(
        seed=123,
        set_system=True,
        set_torch=True):
    """
    Fix random seeds for reproducibility.
    Parameters
    ----------
    seed : int
        Random seed to be set.
    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`
    set_torch : bool
        Whether to set `torch.manual_seed(seed)`
    """
    # set system seed
    if set_system:
        random.seed(seed)
        np.random.seed(seed)

    # set torch seed
    if set_torch:
        torch.manual_seed(seed)

def check_omniglot():
    """
    Check if Omniglot dataset is available.
    """
    if not os.path.isdir("./omniglot_resized"):
        gdd.download_file_from_google_drive(
            file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
            dest_path="./omniglot_resized.zip",
            unzip=True,
        )
    assert os.path.isdir("./omniglot_resized"), "Omniglot dataset is not available! Run `python maml.py --cache` first to download the dataset!"

#########
# TESTS #
#########

# Baseline
class Test_1b(GradedTestCase):
    def setUp(self):
        # self.dataloader_train = _dataloader_helper()
        check_omniglot()
        self.dataloader_train = omniglot.get_omniglot_dataloader(
            split='train',
            batch_size=16,
            num_way=5,
            num_support=1,
            num_query=15,
            num_tasks_per_epoch=240000
        )

        fix_random_seeds()
        self.submission_protonet = submission.ProtoNet(
            0.001, 
            "tests", 
            "cpu"
        )

        fix_random_seeds()
        self.solution_protonet = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.ProtoNet(
            0.001, 
            "tests", 
            "cpu"
        ))

    @graded(timeout=60)
    def test_0(self):
        """1b-0-basic: basic test """

        task_batch = next(iter(self.dataloader_train))

        loss, accuracy_support, accuracy_query = self.submission_protonet._step(task_batch)

        self.assertTrue(loss.shape == torch.Size([]))
        self.assertTrue(type(accuracy_support) == np.float64)
        self.assertTrue(type(accuracy_query) == np.float64)

    @graded(timeout=60, is_hidden=True)
    def test_1(self):
        """1b-1-hidden: hidden test """

        task_batch = next(iter(self.dataloader_train))

        # run solution _step
        fix_random_seeds()
        solution_loss, solution_accuracy_support, solution_accuracy_query = self.solution_protonet._step(task_batch)

        # run submission _step
        fix_random_seeds()
        submission_loss, submission_accuracy_support, submission_accuracy_query = self.submission_protonet._step(task_batch)

        self.assertTrue(torch.allclose(solution_loss, submission_loss, atol=1e-2))
        self.assertTrue(np.allclose(solution_accuracy_support, submission_accuracy_support, atol=1e-2))
        self.assertTrue(np.allclose(solution_accuracy_query, submission_accuracy_query, atol=1e-2))

    @graded(timeout=1, is_hidden=True)
    def test_2(self):
        """1b-2-hidden:  Hidden test case for the Protonet with K = 5."""

        # Load the file containing both labels and predictions
        with open(f'submission/protonet_results_5_5.npy', 'rb') as f:
            loss = np.load(f)
            accuracy_support = np.load(f)
            accuracy_query = np.load(f)

        ### BEGIN_HIDE ###
        ### END_HIDE ###

    @graded(timeout=1, is_hidden=True)
    def test_3(self):
        """1b-3-hidden:  Hidden test case for the Protonet with K = 1."""

        # Load the file containing both labels and predictions
        with open(f'submission/protonet_results_1_5.npy', 'rb') as f:
            loss = np.load(f)
            accuracy_support = np.load(f)
            accuracy_query = np.load(f)

        ### BEGIN_HIDE ###
        ### END_HIDE ###
    

class Test_2a(GradedTestCase):
    def setUp(self):
        check_omniglot()
        self.dataloader_train = omniglot.get_omniglot_dataloader(
            split='train',
            batch_size=16,
            num_way=5,
            num_support=1,
            num_query=15,
            num_tasks_per_epoch=240000,
            num_workers=1,
        )

        self.parameters_keys = ['conv0', 'b0', 'conv1', 'b1', 'conv2', 'b2', 'conv3', 'b3', 'w4', 'b4']
        self.submission_maml = submission.MAML(
            num_outputs=5,
            num_inner_steps=1,
            inner_lr=0.4,
            learn_inner_lrs=False,
            outer_lr=0.001,
            log_dir='./logs/',
            device="cpu"
        )
        self.solution_maml = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.MAML(
            num_outputs=5,
            num_inner_steps=1,
            inner_lr=0.4,
            learn_inner_lrs=False,
            outer_lr=0.001,
            log_dir='./logs/',
            device="cpu"
        ))
    
    @graded(timeout=30)
    def test_0(self):
        """2a-0-basic: check that _inner_loop does not update parameters when train is set to False"""
        fix_random_seeds()

        task_batch = next(iter(self.dataloader_train))
        images_support, labels_support, _, _ = task_batch[0]

        _, _, gradients = self.submission_maml._inner_loop(
            images_support,
            labels_support,
            train=False
        )
        assert all(not grad.requires_grad for grad in gradients), "Gradients should not require grad when train is set to False"

    
    @graded(timeout=30)
    def test_1(self):
        """2a-1-basic: check that _inner_loop does update parameters when train is set to True"""
        fix_random_seeds()

        task_batch = next(iter(self.dataloader_train))
        images_support, labels_support, _, _ = task_batch[0]

        _, _, gradients = self.submission_maml._inner_loop(
            images_support,
            labels_support,
            train=True
        )
        assert all(grad.requires_grad for grad in gradients), "Gradients should require grad when train is set to True"

    
    @graded(timeout=60)
    def test_2(self):
        """2a-2-basic: heck prediction and accuracies shape for _inner_loop"""
        fix_random_seeds()

        task_batch = next(iter(self.dataloader_train))
        images_support, labels_support, _, _ = task_batch[0]

        parameters, accuracies, _ = self.submission_maml._inner_loop(
            images_support,
            labels_support,
            train=True
        )
        self.assertTrue(parameters['conv0'].shape == torch.Size([32, 1, 3, 3]), "conv0 shape is incorrect")
        self.assertTrue(parameters['b0'].shape == torch.Size([32]), "b0 shape is incorrect")
        self.assertTrue(parameters['conv1'].shape == torch.Size([32, 32, 3, 3]), "conv1 shape is incorrect")
        self.assertTrue(parameters['b1'].shape == torch.Size([32]), "b1 shape is incorrect")
        self.assertTrue(parameters['conv2'].shape == torch.Size([32, 32, 3, 3]), "conv2 shape is incorrect")
        self.assertTrue(parameters['b2'].shape == torch.Size([32]), "b2 shape is incorrect")
        self.assertTrue(parameters['conv3'].shape == torch.Size([32, 32, 3, 3]), "conv3 shape is incorrect")
        self.assertTrue(parameters['b3'].shape == torch.Size([32]), "b3 shape is incorrect")
        self.assertTrue(parameters['w4'].shape == torch.Size([5, 32]), "w4 shape is incorrect")
        self.assertTrue(parameters['b4'].shape == torch.Size([5]), "b4 shape is incorrect")
        self.assertTrue(len(accuracies) == 2, "accuracies length is incorrect")

    
    @graded(timeout=90, is_hidden=True)
    def test_3(self):
        """2a-3-hidden: check prediction and accuracies values for initial training steps in _inner_loop"""
        fix_random_seeds()
        target = 3
        can_finish = False
        for i_step, task_batch in enumerate(
                self.dataloader_train,
                start=0
        ):
            for task in task_batch:
                images_support, labels_support, images_query, labels_query = task
                images_support = images_support
                labels_support = labels_support
                images_query = images_query
                labels_query = labels_query
                submission_parameters, submission_accuracies, _ = self.submission_maml._inner_loop(
                    images_support,
                    labels_support,
                    True
                )
                solution_parameters, solution_accuracies, _ = self.solution_maml._inner_loop(
                    images_support,
                    labels_support,
                    True
                )
                if i_step == target:
                    self.assertTrue(torch.allclose(submission_parameters['conv0'], solution_parameters['conv0'], atol=0.8), "values for parameters['conv0'] do not match solutions")
                    self.assertTrue(torch.allclose(submission_parameters['b0'], solution_parameters['b0'], atol=0.8), "values for parameters['b0'] do not match solutions")
                    self.assertTrue(torch.allclose(submission_parameters['conv1'], solution_parameters['conv1'], atol=0.8), "values for parameters['conv1'] do not match solutions")
                    self.assertTrue(torch.allclose(submission_parameters['b1'], solution_parameters['b1'], atol=0.8), "values for parameters['b1'] do not match solutions")
                    self.assertTrue(torch.allclose(submission_parameters['conv2'], solution_parameters['conv2'], atol=0.8), "values for parameters['conv2'] do not match solutions")
                    self.assertTrue(torch.allclose(submission_parameters['b2'], solution_parameters['b2'], atol=0.8), "values for parameters['b2'] do not match solutions")
                    self.assertTrue(torch.allclose(submission_parameters['conv3'], solution_parameters['conv3'], atol=0.8), "values for parameters['conv3'] do not match solutions")
                    self.assertTrue(torch.allclose(submission_parameters['b3'], solution_parameters['b3'], atol=0.8), "values for parameters['b3'] do not match solutions")
                    self.assertTrue(torch.allclose(submission_parameters['w4'], solution_parameters['w4'], atol=0.8), "values for parameters['w4'] do not match solutions")
                    self.assertTrue(torch.allclose(submission_parameters['b4'], solution_parameters['b4'], atol=0.8), "values for parameters['b4'] do not match solutions")
                    self.assertTrue(np.allclose(submission_accuracies, solution_accuracies, atol=0.8), "accuracies do not match solution accuracies for _inner_loop")
                    can_finish = True
            if can_finish:
                break

class Test_2b(GradedTestCase):
    def setUp(self):
        check_omniglot()
        self.dataloader_train = omniglot.get_omniglot_dataloader(
            split='train',
            batch_size=8,
            num_way=5,
            num_support=1,
            num_query=15,
            num_tasks_per_epoch=240000,
            num_workers=1,
        )
        self.parameters_keys = ['conv0', 'b0', 'conv1', 'b1', 'conv2', 'b2', 'conv3', 'b3', 'w4', 'b4']
        self.submission_maml = submission.MAML(
            num_outputs=5,
            num_inner_steps=1,
            inner_lr=0.4,
            learn_inner_lrs=False,
            outer_lr=0.001,
            log_dir='./logs/',
            device="cpu"
        )
        self.solution_maml = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.MAML(
            num_outputs=5,
            num_inner_steps=1,
            inner_lr=0.4,
            learn_inner_lrs=False,
            outer_lr=0.001,
            log_dir='./logs/',
            device="cpu"
        ))
    
    @graded(timeout=60)
    def test_0(self):
        """2b-0-basic: check shapes are correct for _outer_step"""
        fix_random_seeds()

        task_batch = next(iter(self.dataloader_train))

        self.submission_maml._optimizer.zero_grad()
        outer_loss, accuracies_support, accuracy_query = (
            self.submission_maml._outer_step(task_batch, train=True)
        )
        self.assertTrue(outer_loss.shape == torch.Size([]))
        self.assertTrue(accuracies_support.shape == (2,))
        self.assertTrue(type(accuracy_query) == np.float64)

    
    @graded(timeout=60, is_hidden=True)
    def test_1(self):
        """2b-1-hidden: check values for _outer_step during initial training"""
        fix_random_seeds()

        task_batch = next(iter(self.dataloader_train))
        
        self.submission_maml._optimizer.zero_grad()
        self.solution_maml._optimizer.zero_grad()
        submission_outer_loss, submission_accuracies_support, submission_accuracy_query = (
            self.submission_maml._outer_step(task_batch, train=True)
        )
        solution_outer_loss, solution_accuracies_support, solution_accuracy_query = (
            self.solution_maml._outer_step(task_batch, train=True)
        )
        
        self.assertTrue(torch.allclose(submission_outer_loss, solution_outer_loss, atol=0.25), "outer_loss do not match solutions")
        self.assertTrue(np.allclose(submission_accuracies_support, solution_accuracies_support, atol=0.25), "accuracies_support do not match solutions")
        self.assertTrue(np.allclose(submission_accuracy_query, solution_accuracy_query, atol=0.25), "accuracy_query do not match solutions")

    @graded(timeout=1, is_hidden=True)
    def test_2(self):
        """2b-2-hidden:  Hidden test case for the default MAML."""

        # Load the file containing both labels and predictions
        with open(f'submission/maml_results_1_5_1_0.4_False.npy', 'rb') as f:
            loss = np.load(f)
            accuracy_pre_adapt_support = np.load(f)
            accuracy_post_adapt_support = np.load(f)
            accuracy_post_adapt_query = np.load(f)

        ### BEGIN_HIDE ###
        ### END_HIDE ###

    @graded(timeout=1, is_hidden=True)
    def test_3(self):
        """2b-3-hidden:  Hidden test case for the MAML with 0.04 learning rate."""

        # Load the file containing both labels and predictions
        with open(f'submission/maml_results_1_5_1_0.04_False.npy', 'rb') as f:
            loss = np.load(f)
            accuracy_pre_adapt_support = np.load(f)
            accuracy_post_adapt_support = np.load(f)
            accuracy_post_adapt_query = np.load(f)

        ### BEGIN_HIDE ###
        ### END_HIDE ###

    @graded(timeout=1, is_hidden=True)
    def test_4(self):
        """2b-4-hidden:  Hidden test case for the MAML with 0.04 learning rate and 5 inner steps."""

        # Load the file containing both labels and predictions
        with open(f'submission/maml_results_1_5_5_0.04_False.npy', 'rb') as f:
            loss = np.load(f)
            accuracy_pre_adapt_support = np.load(f)
            accuracy_post_adapt_support = np.load(f)
            accuracy_post_adapt_query = np.load(f)

        ### BEGIN_HIDE ###
        ### END_HIDE ###

    @graded(timeout=1, is_hidden=True)
    def test_5(self):
        """2b-5-hidden:  Hidden test case for the MAML with 0.04 learning rate and 5 inner steps."""

        # Load the file containing both labels and predictions
        with open(f'submission/maml_results_1_5_1_0.4_True.npy', 'rb') as f:
            loss = np.load(f)
            accuracy_pre_adapt_support = np.load(f)
            accuracy_post_adapt_support = np.load(f)
            accuracy_post_adapt_query = np.load(f)

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
