import jax.numpy as jnp
import flax.linen as nn

class ImpalaBlock(nn.Module):
    out_ch: int

    @nn.compact
    def __call__(self, x):
        # ResNet block:
        # (a) ReLU activation followed by BatchNorm
        y = nn.relu(x)
        y = nn.BatchNorm(use_running_average=True)(y)
        # (b) Convolutional layer: 3×3 kernel, stride 1
        y = nn.Conv(self.out_ch, kernel_size=(3,3), strides=(1,1), padding='SAME')(y)
        # Residual connection: add input x
        return x + y

class ImpalaCNN(nn.Module):
    """
    Impala CNN with three stacks of channels (64, 64, 128). Each stack:
      (a) BatchNorm
      (b) Conv 3×3, stride 1
      (c) MaxPool 3×3, stride 2
      (d) Two ResNet blocks (ImpalaBlock)
    Input: [B, 63, 63, 3]
    Output: [B, 8*8*128] flattened embedding vector of size 8192
    """
    channels: tuple = (64, 64, 128)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for ch in self.channels:
            # (a) Batch normalization
            x = nn.BatchNorm(use_running_average=True)(x)
            # (b) Convolutional layer: 3×3 kernel, stride 1
            x = nn.Conv(ch, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)
            # (c) Max pooling: 3×3 window, stride 2
            x = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding='SAME')
            # (d) Two ResNet blocks
            x = ImpalaBlock(ch)(x)
            x = ImpalaBlock(ch)(x)
        # Final activation before flattening
        x = nn.relu(x)
        # Flatten spatial dims into embedding vector
        B, H, W, C = x.shape
        return x.reshape((B, H * W * C))