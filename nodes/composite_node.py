"""
Composite Image Node for ComfyUI.

This node composites a source image onto a destination image at specified relative positions.
"""

from typing import Tuple
import torch
from PIL import Image

from .base import BaseImageNode


class CompositeImageNode(BaseImageNode):
    """
    A ComfyUI node that composites a source image onto a destination image.

    The position is specified as relative coordinates (0.0 to 1.0), with alignment options.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return cls.create_input_types(
            source_image=cls.INPUT_TYPE_IMAGE,
            destination_image=cls.INPUT_TYPE_IMAGE,
            x_position=(
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            y_position=(
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            alignment=(["top_left", "center"], {"default": "top_left"}),
        )

    RETURN_TYPES = BaseImageNode.RETURN_TYPES_IMAGE
    FUNCTION = "composite_images"
    CATEGORY = BaseImageNode.CATEGORY_COMPOSITE

    def composite_images(
        self,
        source_image: torch.Tensor,
        destination_image: torch.Tensor,
        x_position: float,
        y_position: float,
        alignment: str,
    ) -> Tuple[torch.Tensor]:
        """
        Composite the source image onto the destination image at the specified position.

        Args:
            source_image: Source image tensor in BHWC format.
            destination_image: Destination image tensor in BHWC format.
            x_position: Relative x position (0.0 = left edge, 1.0 = right edge).
            y_position: Relative y position (0.0 = top edge, 1.0 = bottom edge).
            alignment: Alignment mode - "top_left" (corner) or "center".

        Returns:
            A tuple containing the composited image tensor.

        Raises:
            InvalidInputError: If input parameters are invalid.
        """
        # Validate inputs
        self.validate_image_tensor(source_image, "source_image")
        self.validate_image_tensor(destination_image, "destination_image")
        self.validate_numeric_input(x_position, "x_position", min_val=0.0, max_val=1.0)
        self.validate_numeric_input(y_position, "y_position", min_val=0.0, max_val=1.0)
        self.validate_string_input(alignment, "alignment", ["top_left", "center"])

        # Convert tensors to PIL Images
        src_pil = self.tensor_to_pil(source_image)
        dst_pil = self.tensor_to_pil(destination_image)

        # Calculate position
        dst_width, dst_height = dst_pil.size
        src_width, src_height = src_pil.size

        if alignment == "top_left":
            x = int((dst_width - src_width) * x_position)
            y = int((dst_height - src_height) * y_position)
        elif alignment == "center":
            x = int(x_position * dst_width - src_width / 2)
            y = int(y_position * dst_height - src_height / 2)

        # Ensure position is within bounds
        x = max(0, min(x, dst_width - src_width))
        y = max(0, min(y, dst_height - src_height))

        # Composite images (handle transparency if RGBA)
        dst_pil.paste(src_pil, (x, y), src_pil if src_pil.mode == "RGBA" else None)

        # Convert back to tensor
        comp_tensor = self.pil_to_tensor(dst_pil)

        return (comp_tensor,)
