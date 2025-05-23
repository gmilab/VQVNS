import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence

class SimVQQuantizer(nn.Module):
    """
    SimVQQuantizer implements the SimVQ reparameterization for vector quantization.
    Instead of using EMA updates to learn the codebook, it reparameterizes the codebook
    as a fixed embedding matrix C and a learnable latent basis W. The effective codebook is given by:
        C_eff = C @ W
    Only W is updated via gradient descent while C remains fixed.
    
    Args:
        spatial_dims (int): Number of spatial dimensions of the input (e.g., 2 for images).
        num_embeddings (int): Number of codebook vectors.
        embedding_dim (int): Dimensionality of each codebook vector.
        commitment_cost (float): Scaling factor for the commitment loss.
        embedding_init (str): Initialization method for the codebook ("normal" or "kaiming_uniform").
    """
    def __init__(
        self,
        spatial_dims: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        embedding_init: str = "normal",
        legacy: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.legacy = legacy # This is from the SimVQ paper where the author made a mistake about which term is beta weighted
        
        # Define the codebook C as an embedding layer; its weights will remain fixed.
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        if embedding_init == "normal":
            # Default initialization in nn.Embedding is used.
            pass
        elif embedding_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.codebook.weight, mode="fan_in", nonlinearity="linear")
        self.codebook.weight.requires_grad = False  # Freeze C
        
        # Define the learnable latent basis W as a d x d matrix.
        # W is initialized to the identity matrix.
        self.W = nn.Parameter(torch.eye(embedding_dim))
        
        # Pre-calculate permutation orders for reshaping inputs.
        # This converts from channel-first to channel-last for flatting.
        self.flatten_permutation = [0] + list(range(2, spatial_dims + 2)) + [1]
        self.quantization_permutation: Sequence[int] = [0, spatial_dims + 1] + list(range(1, spatial_dims + 1))
    
    def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects the input tensor into the quantized space using the reparameterized codebook.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W, ...] (depending on spatial_dims).
            
        Returns:
            flat_input (torch.Tensor): Flattened input of shape [B * (H*W*...), C].
            encodings (torch.Tensor): One-hot encoding of quantization indices, shape [B * (H*W*...), num_embeddings].
            encoding_indices (torch.Tensor): Quantization indices reshaped to match the spatial dimensions.
        """
        with torch.cuda.amp.autocast(enabled=False):
            original_shape = list(inputs.shape)
            # Permute to channel-last format and flatten spatial dimensions.
            flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)
            
            # Compute the effective codebook: C_eff = C @ W (shape: [num_embeddings, embedding_dim])
            effective_weight = self.codebook.weight @ self.W
            
            # Calculate Euclidean distances between flat_input and each effective code vector.
            distances = (
                (flat_input ** 2).sum(dim=1, keepdim=True)
                + (effective_weight.t() ** 2).sum(dim=0, keepdim=True)
                - 2 * torch.mm(flat_input, effective_weight.t())
            )
            
            # Get the index of the nearest code vector for each input.
            encoding_indices = torch.max(-distances, dim=1)[1]
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            
            # Reshape encoding_indices to the original spatial layout (excluding channel dimension).
            spatial_shape = original_shape[0:1] + original_shape[2:]
            encoding_indices = encoding_indices.view(spatial_shape)
            
            return flat_input, encodings, encoding_indices
        
    def embed(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """
        Looks up and transforms the quantized code vectors using the latent basis.
        
        Args:
            encoding_indices (torch.Tensor): Tensor containing quantization indices.
            
        Returns:
            torch.Tensor: Quantized tensor transformed by W and permuted back to channel-first format.
        """
        with torch.cuda.amp.autocast(enabled=False):
            # Lookup codebook vectors from the fixed embedding.
            code_vectors = self.codebook(encoding_indices)  # Shape: [B, H, W, ..., embedding_dim]
            # Apply the learnable transformation: effective code = code_vectors @ W.
            quantized = torch.matmul(code_vectors, self.W)
            # Permute back to channel-first format.
            quantized = quantized.permute(self.quantization_permutation).contiguous()
            return quantized
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SimVQQuantizer.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W, ...].
        
        Returns:
            quantized (torch.Tensor): Quantized tensor (with gradients passed via the straight-through estimator).
            loss (torch.Tensor): Commitment loss.
            encoding_indices (torch.Tensor): Quantization indices.
        """
        flat_input, encodings, encoding_indices = self.quantize(inputs)
        quantized = self.embed(encoding_indices)
        
        # Compute the commitment loss between the inputs and the quantized outputs.
        if self.legacy:
            loss = F.mse_loss(quantized.detach(), inputs) + self.commitment_cost * F.mse_loss(quantized, inputs.detach())
        else:
            loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs) + F.mse_loss(quantized, inputs.detach())

        # Apply the straight-through estimator: pass gradients from quantized to inputs.
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices
    

class SimVQQuantizerWrapper(torch.nn.Module):
    """
    Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
    the quantization in their own class.

    Args:
        quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
            based quantized representation.
    """

    def __init__(self, quantizer: SimVQQuantizer):
        super().__init__()

        self.quantizer: SimVQQuantizer = quantizer

        self.perplexity: torch.Tensor = torch.rand(1)

        self.w_rank: torch.Tensor = torch.rand(1)

        self.w_frobenius_norm: torch.Tensor = torch.rand(1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, loss, encoding_indices = self.quantizer(inputs)
        # Perplexity calculations
        avg_probs = (
            torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
            .float()
            .div(encoding_indices.numel())
        )

        self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        #Calculate the rank and frobenius norm of the weight matrix
        self.w_rank = torch.linalg.matrix_rank(self.quantizer.W)
        self.w_frobenius_norm = torch.linalg.norm(self.quantizer.W, ord='fro')

        return loss, quantized

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        output = self.quantizer(encodings)
        encoding_indices: torch.Tensor = output[2]
        return encoding_indices
