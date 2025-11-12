"""
Scale Image Node for ComfyUI.

This node scales an input image to a specified width or height while maintaining aspect ratio.
"""

from typing import Tuple
import torch
from PIL import Image

from .base import BaseImageNode


class ScaleImageNode(BaseImageNode):
    """
    A ComfyUI node that scales an image to a specified width or height.

    The scaling maintains the aspect ratio of the original image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return cls.create_input_types(
            image=cls.INPUT_TYPE_IMAGE,
            target_dimension=(
                "FLOAT",
                {"default": 512.0, "min": 1.0, "max": 4096.0, "step": 1.0},
            ),
            dimension_type=(["width", "height"], {"default": "width"}),
        )

    RETURN_TYPES = BaseImageNode.RETURN_TYPES_IMAGE
    FUNCTION = "scale_image"
    CATEGORY = BaseImageNode.CATEGORY_UTILITY

    def scale_image(
        self, image: torch.Tensor, target_dimension: float, dimension_type: str
    ) -> Tuple[torch.Tensor]:
        """
        Scale the input image to the target dimension.

        Args:
            image: Input image tensor in BHWC format.
            target_dimension: The target width or height value.
            dimension_type: Whether to scale by 'width' or 'height'.

        Returns:
            A tuple containing the scaled image tensor.

        Raises:
            InvalidInputError: If input parameters are invalid.
        """
        # Validate inputs
        self.validate_image_tensor(image, "image")
        self.validate_numeric_input(
            target_dimension, "target_dimension", min_val=1.0, max_val=4096.0
        )
        self.validate_string_input(
            dimension_type, "dimension_type", ["width", "height"]
        )

        # Convert tensor to PIL Image
        pil_image = self.tensor_to_pil(image)
        original_width, original_height = pil_image.size

        # Calculate new dimensions
        if dimension_type == "width":
            new_width = int(target_dimension)
            new_height = int((new_width / original_width) * original_height)
        else:  # height
            new_height = int(target_dimension)
            new_width = int((new_height / original_height) * original_width)

        # Ensure minimum dimensions
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Resize the image using high-quality Lanczos resampling
        scaled_pil = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert back to tensor
        scaled_tensor = self.pil_to_tensor(scaled_pil)

        return (scaled_tensor,)
