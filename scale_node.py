"""
Scale Image Node for ComfyUI.

This node scales an input image to a specified width or height while maintaining aspect ratio.
"""

import torch
from PIL import Image
import numpy as np


class ScaleImageNode:
    """
    A ComfyUI node that scales an image to a specified width or height.

    The scaling maintains the aspect ratio of the original image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_dimension": (
                    "FLOAT",
                    {"default": 512.0, "min": 1.0, "max": 4096.0},
                ),
                "dimension_type": (["width", "height"], {"default": "width"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale_image"
    CATEGORY = "image/processing"

    def scale_image(
        self, image: torch.Tensor, target_dimension: float, dimension_type: str
    ) -> tuple:
        """
        Scale the input image to the target dimension.

        Args:
            image: Input image tensor of shape (batch, height, width, channels).
            target_dimension: The target width or height value.
            dimension_type: Whether to scale by 'width' or 'height'.

        Returns:
            A tuple containing the scaled image tensor.
        """
        # Convert torch tensor to PIL Image
        # Assuming image is in range [0, 1], convert to [0, 255]
        image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        original_width, original_height = pil_image.size

        if dimension_type == "width":
            new_width = int(target_dimension)
            new_height = int((new_width / original_width) * original_height)
        else:  # height
            new_height = int(target_dimension)
            new_width = int((new_height / original_height) * original_width)

        # Resize the image
        scaled_pil = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert back to torch tensor
        scaled_np = np.array(scaled_pil).astype(np.float32) / 255.0
        scaled_tensor = torch.from_numpy(scaled_np).unsqueeze(0)

        return (scaled_tensor,)
