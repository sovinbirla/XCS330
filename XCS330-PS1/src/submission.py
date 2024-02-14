import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 embedding_sharing=True):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_sharing = embedding_sharing

        if embedding_sharing:
            self.U, self.Q = self.init_shared_user_and_item_embeddings(num_users, num_items, embedding_dim)
        else:
            self.U_reg, self.Q_reg, self.U_fact, self.Q_fact = self.init_separate_user_and_item_embeddings(num_users, num_items, embedding_dim)

        self.A, self.B = self.init_user_and_item_bias(num_users, num_items)
        self.mlp_layers = self.init_mlp_layers(layer_sizes)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        if self.embedding_sharing:
            predictions, score = self.forward_with_embedding_sharing(user_ids, item_ids)
        else:
            predictions, score = self.forward_without_embedding_sharing(user_ids, item_ids)

        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score
    
    def init_shared_user_and_item_embeddings(self, num_users, num_items, embedding_dim):
        """
        Initializes shared user and item embeddings
        used in both factorization and regression tasks

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.
            

        Returns
        -------

        U: ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q: ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        """
        U = Q = None
        

        # print("num_users:", num_users)
        # print("num_items:", num_items)

        U = ScaledEmbedding(num_users, embedding_dim=embedding_dim)
        Q = ScaledEmbedding(num_items, embedding_dim=embedding_dim)



        ### END CODE HERE ###
        return U, Q
    
    def init_separate_user_and_item_embeddings(self, num_users, num_items, embedding_dim):
        """
        Initializes separate user and item embeddings
        where one will be used for factorization (ie _fact) and 
        other for regression tasks (ie _reg)

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.
            

        Returns
        -------

        U_reg: first ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_reg: first ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        U_fact: second ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_fact: second ScaledEmbedding layer for items  
            nn.Embedding of shape (num_items, embedding_dim)

        Note: Order does matter here! Please declare the layers in the order
        they are returned.
        """
        U_reg = Q_reg = U_fact = Q_fact = None
        ### START CODE HERE ###
        
        U_reg = ScaledEmbedding(num_users, embedding_dim=embedding_dim)
        Q_reg = ScaledEmbedding(num_items, embedding_dim=embedding_dim)
        U_fact = ScaledEmbedding(num_users, embedding_dim=embedding_dim)
        Q_fact = ScaledEmbedding(num_items, embedding_dim=embedding_dim)

        ### END CODE HERE ###
        return U_reg, Q_reg, U_fact, Q_fact
    
    def init_user_and_item_bias(self, num_users, num_items):
        """
        Initializes user and item bias terms

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.

        Returns
        -------

        A: ZeroEmbedding layer for users
            nn.Embedding of shape (num_users, 1)
        B: ZeroEmbedding layer for items
            nn.Embedding of shape (num_items, 1)
        """
        A = B = None
        ### START CODE HERE ###

        A = ZeroEmbedding(num_users, embedding_dim=1)
        B = ZeroEmbedding(num_items, embedding_dim=1)

        ### END CODE HERE ###
        return A, B
    
    def init_mlp_layers(self, layer_sizes):
        """
        Initializes MLP layer for regression task

        Parameters
        ----------

        layer_sizes: list
            List of layer sizes to for the regression network.

        Returns
        -------

        mlp_layers: nn.ModuleList
            MLP network containing Linear and ReLU layers
        """
        mlp_layers = None
        ### START CODE HERE ###

        #should have used nn.Sequential(), but it works

    
        mlp_layers = nn.ModuleList([])

        for i in range(len(layer_sizes) - 1):
            mlp_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1] ))
            mlp_layers.append(nn.ReLU())

        mlp_layers.append(nn.Linear(layer_sizes[-1], 1))


        ### END CODE HERE ###
        return mlp_layers

    def forward_with_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None
        ### START CODE HERE ###
        

        # print(user_ids.shape, user_ids[:5])
        # print(item_ids.shape, item_ids[:5])

        U = self.U(user_ids)
        Q = self.Q(item_ids)
        A= self.A(user_ids)
        B= self.B(item_ids)


        # print("U shape: ", U.shape)
        # print("Q shape: ", Q.shape)
        # print("A shape: ", A.shape)
        # print("B shape: ", B.shape)
        
        predictions =  torch.unsqueeze(torch.sum(U* Q, dim=1), dim=1) + A + B
        mlp_input = torch.cat((U, Q, U* Q), 1)

        # print("input:" , mlp_input.size())

        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            # print("input:" , mlp_input.size())

        # print("predictions:" , predictions.shape)
        score = mlp_input
        # print("score:" , score.shape)
        ### END CODE HERE ###
        return torch.squeeze(predictions), torch.squeeze(score)
    
    def forward_without_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None
        ### START CODE HERE ###
        U_reg = self.U_reg(user_ids)
        U_fact = self.U_fact(user_ids)
        Q_reg = self.Q_reg(item_ids)
        Q_fact = self.Q_fact(item_ids)
        A= self.A(user_ids)
        B= self.B(item_ids)

        # print("U_reg shape: ", U_reg.shape)
        # print("U_fact shape: ", U_fact.shape)
        # print("Q_reg shape: ", Q_reg.shape)
        # print("Q_fact shape: ", Q_fact.shape)
        # print("A shape: ", A.shape)
        # print("B shape: ", B.shape)


        predictions = torch.unsqueeze(torch.sum(U_fact * Q_fact, dim=1), dim=1) + A +  B

        score = torch.cat((U_reg, Q_reg, U_reg * Q_reg), 1)

        # print("input:" , score.size())

        for layer in self.mlp_layers:
            score = layer(score)
        #     print("input:" , score.size())

        # print("predictions:" , predictions.shape)


        return torch.squeeze(predictions), torch.squeeze(score)