import argparse
import torch
import numpy as np

from dataset import get_movielens_dataset
from evaluation import mrr_score, mse_score
from submission import MultiTaskNet
from multitask import MultitaskModel
from utils import fix_random_seeds
from torch.utils.tensorboard import SummaryWriter

def main(config):
    print(config)

    fix_random_seeds()
    
    use_gpu = False
    if config.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        use_gpu = True
    elif config.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        use_gpu = True
    else:
        device = torch.device("cpu")
    print("Using device: ", device)

    writer = SummaryWriter(config.logdir)

    dataset = get_movielens_dataset(variant='100K')
    train, test = dataset.random_train_test_split(test_fraction=config.test_fraction)

    net = MultiTaskNet(train.num_users,
                       train.num_items,
                       embedding_sharing=config.shared_embeddings)
    model = MultitaskModel(interactions=train,
                           representation=net,
                           factorization_weight=config.factorization_weight,
                           regression_weight=config.regression_weight,
                           use_gpu=use_gpu)
    
    if(config.compile == True):
        try:
            net = torch.compile(net, backend=config.backend)
            print(f"MultiTaskNet compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    if config.debug:
        print("==========================")
        print("Starting Multitask Training")
        print("NOTE: You can run `tensorboard --logdir=run` to see training in real time")
        print(f"factorization_weight: {config.factorization_weight}")
        print(f"regression_weight: {config.regression_weight}")
        print(f"shared_embeddings: {config.shared_embeddings}")
        print("==========================")
    for epoch in range(config.epochs):
        factorization_loss, score_loss, joint_loss = model.fit(train)
        mrr = mrr_score(model, test, train)
        mse = mse_score(model, test)
        if config.debug and epoch % 10 == 0:
            print(f"Running epoch {epoch} / {config.epochs}")
            print(f"training/Factorization Loss: {factorization_loss}")
            print(f"training/MSE: {score_loss}")
            print(f"training/Joint Loss: {joint_loss}")
            print(f"eval/Mean Reciprocal Rank: {mrr}")
            print(f"eval/MSE: {mse}")
            print("==========================")
        writer.add_scalar('training/Factorization Loss', factorization_loss, epoch)
        writer.add_scalar('training/MSE', score_loss, epoch)
        writer.add_scalar('training/Joint Loss', joint_loss, epoch)
        writer.add_scalar('eval/Mean Reciprocal Rank', mrr, epoch)
        writer.add_scalar('eval/MSE', mse, epoch)
    
    if config.shared_embeddings and config.factorization_weight == 0.99 and config.regression_weight == 0.01:
        file_name = "experiment_1"
    elif config.shared_embeddings and config.factorization_weight == 0.5 and config.regression_weight == 0.5:
        file_name = "experiment_2"
    elif not config.shared_embeddings and config.factorization_weight == 0.5 and config.regression_weight == 0.5:
        file_name = "experiment_3"
    elif not config.shared_embeddings and config.factorization_weight == 0.99 and config.regression_weight == 0.01:
        file_name = "experiment_4"
    else:
        file_name = "experiment"

    with open(f'./{file_name}.npy', 'wb') as f:
        np.save(f, factorization_loss)
        np.save(f, score_loss)
        np.save(f, joint_loss)
        np.save(f, mrr)
        np.save(f, mse)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_fraction', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--factorization_weight', type=float, default=0.995)
    parser.add_argument('--regression_weight', type=float, default=0.005)
    parser.add_argument('--shared_embeddings', default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--logdir', type=str,
                        default='run/shared=True_LF=0.99_LR=0.01')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--cache', action='store_true')

    args = parser.parse_args()

    if args.cache == True:
        # Download Omniglot Dataset
        _ = get_movielens_dataset(variant='100K')
    else:
        main(args)