"""
Composite Image Node for ComfyUI.

This node composites a source image onto a destination image at specified relative positions.
"""

import torch
from PIL import Image
import numpy as np


class CompositeImageNode:
    """
    A ComfyUI node that composites a source image onto a destination image.

    The position is specified as relative coordinates (0.0 to 1.0), with alignment options.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "destination_image": ("IMAGE",),
                "x_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alignment": (["top_left", "center"], {"default": "top_left"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite_images"
    CATEGORY = "image/processing"

    def composite_images(
        self,
        source_image: torch.Tensor,
        destination_image: torch.Tensor,
        x_position: float,
        y_position: float,
        alignment: str,
    ) -> tuple:
        """
        Composite the source image onto the destination image at the specified position.

        Args:
            source_image: Source image tensor.
            destination_image: Destination image tensor.
            x_position: Relative x position (0.0 = left, 1.0 = right).
            y_position: Relative y position (0.0 = top, 1.0 = bottom).
            alignment: Alignment mode - "top_left" or "center".

        Returns:
            A tuple containing the composited image tensor.
        """
        # Convert tensors to PIL Images
        src_np = (source_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        dst_np = (destination_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        src_pil = Image.fromarray(src_np)
        dst_pil = Image.fromarray(dst_np)

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

        # Composite images
        dst_pil.paste(src_pil, (x, y), src_pil if src_pil.mode == "RGBA" else None)

        # Convert back to tensor
        comp_np = np.array(dst_pil).astype(np.float32) / 255.0
        comp_tensor = torch.from_numpy(comp_np).unsqueeze(0)

        return (comp_tensor,)
