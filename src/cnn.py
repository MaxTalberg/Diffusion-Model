import torch
import numpy as np
import torch.nn as nn

class CNNBlock(nn.Module):
    """
    A convolutional block that forms the basic building unit of a CNN. It comprises a convolutional layer followed by layer normalization and an activation function.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    expected_shape : tuple of int
        The expected spatial dimensions (H, W) of the output tensor from the convolutional layer.
    act : nn.Module, optional
        The activation function to use. Defaults to `nn.GELU`.
    kernel_size : int, optional
        Size of the convolving kernel. Defaults to 7.

    Attributes
    ----------
    net : nn.Sequential
        The sequential model containing the convolutional layer, layer normalization, and activation function.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm((out_channels, *expected_shape)),
            act()
        )

    def forward(self, x):
        """
        Forward pass of the CNNBlock.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the CNNBlock.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the convolution, normalization, and activation function.
        """
        return self.net(x)
    
class CNN(nn.Module):
    """
    A convolutional neural network model that incorporates a series of `CNNBlock` layers, followed by a final convolutional layer. 
    The model also includes a mechanism to embed temporal information into the input tensor, which is useful for tasks involving time-dependent data.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    expected_shape : tuple of int, optional
        The expected spatial dimensions (H, W) of the output tensor from each convolutional block. Defaults to (28, 28).
    n_hidden : tuple of int, optional
        A sequence defining the number of channels for the hidden layers in the CNN. Defaults to (64, 128, 64).
    kernel_size : int, optional
        Size of the convolving kernel for the convolutional blocks. Defaults to 7.
    last_kernel_size : int, optional
        Size of the convolving kernel for the final convolutional layer. Defaults to 3.
    time_embeddings : int, optional
        Dimensionality of the time embeddings. Defaults to 16.
    act : nn.Module, optional
        The activation function to use in the convolutional blocks and time embedding layers. Defaults to `nn.GELU`.

    Attributes
    ----------
    blocks : nn.ModuleList
        A list of convolutional blocks and the final convolutional layer that make up the CNN.
    time_embed : nn.Sequential
        A sequential model to process the time encoding and embed it into the CNN.
    frequencies : torch.Tensor
        The frequencies used for the sinusoidal time encoding.

    """
    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16,
        act=nn.GELU,
    ) -> None:
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        ## This part is literally just to put the single scalar "t" into the CNN
        ## in a nice, high-dimensional way:
        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128), act(),
            nn.Linear(128, 128), act(),
            nn.Linear(128, 128), act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encodes temporal information into a tensor using sinusoidal functions, to be added to the network's latent space.

        Parameters
        ----------
        t : torch.Tensor
            The input tensor representing time or timestep information.

        Returns
        -------
        torch.Tensor
            The time-encoded tensor ready to be added to the CNN's latent space.
        """
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model with time encoding.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the CNN, typically an image or batch of images.
        t : torch.Tensor
            The tensor representing time or timestep information for each input in the batch.

        Returns
        -------
        torch.Tensor
            The output tensor of the CNN after processing the input
        """
        # Shapes of input:
        #    x: (batch, chan, height, width)
        #    t: (batch,)

        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t)
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed
    