"""
Base classes and utilities for ComfyUI Image Edit Nodes.

This module provides the foundation for all image processing nodes,
including common utilities for tensor/PIL conversions, validation, and error handling.
"""

import abc
from typing import Any, Dict, Optional, Tuple, Union
import torch
from PIL import Image
import numpy as np


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""

    pass


class InvalidInputError(ImageProcessingError):
    """Raised when input parameters are invalid."""

    pass


class ImageConversionError(ImageProcessingError):
    """Raised when image conversion fails."""

    pass


class BaseImageNode(abc.ABC):
    """
    Abstract base class for all ComfyUI image processing nodes.

    This class provides common utilities for:
    - Converting between ComfyUI tensors and PIL Images
    - Input validation and error handling
    - Consistent node interface patterns

    All image processing nodes should inherit from this class.
    """

    # Default categories for different node types
    CATEGORY_TRANSFORM = "image/transform"
    CATEGORY_ADJUSTMENT = "image/adjustment"
    CATEGORY_COLOR = "image/color"
    CATEGORY_FILTER = "image/filter"
    CATEGORY_UTILITY = "image/utility"
    CATEGORY_COMPOSITE = "image/composite"

    # Common return types
    RETURN_TYPES_IMAGE = ("IMAGE",)
    RETURN_TYPES_MASK = ("MASK",)
    RETURN_TYPES_IMAGE_MASK = ("IMAGE", "MASK")

    # Common input types
    INPUT_TYPE_IMAGE = ("IMAGE",)
    INPUT_TYPE_MASK = ("MASK",)

    @classmethod
    @abc.abstractmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining the node's input parameters.
        """
        pass

    @property
    @abc.abstractmethod
    def RETURN_TYPES(self) -> Tuple[str, ...]:
        """Define the output types for this node."""
        pass

    @property
    @abc.abstractmethod
    def FUNCTION(self) -> str:
        """Define the main processing function name."""
        pass

    @property
    @abc.abstractmethod
    def CATEGORY(self) -> str:
        """Define the category this node belongs to."""
        pass

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convert a ComfyUI tensor to a PIL Image.

        Args:
            tensor: Input tensor in BHWC format (batch, height, width, channels)

        Returns:
            PIL Image object

        Raises:
            ImageConversionError: If conversion fails
        """
        try:
            # Remove batch dimension if present (assume batch size 1)
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)

            # Convert to numpy array and scale to 0-255
            # Assuming tensor is in range [0, 1]
            np_array = (tensor.cpu().numpy() * 255).astype(np.uint8)

            # Create PIL Image
            if tensor.shape[-1] == 3:  # RGB
                pil_image = Image.fromarray(np_array, mode="RGB")
            elif tensor.shape[-1] == 4:  # RGBA
                pil_image = Image.fromarray(np_array, mode="RGBA")
            elif tensor.shape[-1] == 1:  # Grayscale
                pil_image = Image.fromarray(np_array.squeeze(-1), mode="L")
            else:
                raise ImageConversionError(
                    f"Unsupported number of channels: {tensor.shape[-1]}"
                )

            return pil_image

        except Exception as e:
            raise ImageConversionError(
                f"Failed to convert tensor to PIL Image: {e}"
            ) from e

    @staticmethod
    def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
        """
        Convert a PIL Image to a ComfyUI tensor.

        Args:
            pil_image: Input PIL Image

        Returns:
            Tensor in BHWC format (batch, height, width, channels)

        Raises:
            ImageConversionError: If conversion fails
        """
        try:
            # Handle different modes appropriately
            if pil_image.mode == "RGB":
                np_array = np.array(pil_image)
            elif pil_image.mode == "RGBA":
                np_array = np.array(pil_image)
            elif pil_image.mode == "L":
                # Keep grayscale as single channel
                np_array = np.array(pil_image)
                np_array = np_array[..., np.newaxis]  # Add channel dimension
            else:
                # Convert other modes to RGB
                pil_image = pil_image.convert("RGB")
                np_array = np.array(pil_image)

            # Convert to float and scale to [0, 1]
            tensor = torch.from_numpy(np_array.astype(np.float32) / 255.0)

            # Add batch dimension
            tensor = tensor.unsqueeze(0)

            return tensor

        except Exception as e:
            raise ImageConversionError(
                f"Failed to convert PIL Image to tensor: {e}"
            ) from e

    @staticmethod
    def validate_image_tensor(tensor: torch.Tensor, name: str = "image") -> None:
        """
        Validate that a tensor represents a valid image.

        Args:
            tensor: Input tensor to validate
            name: Name of the tensor for error messages

        Raises:
            InvalidInputError: If tensor is invalid
        """
        if not isinstance(tensor, torch.Tensor):
            raise InvalidInputError(
                f"{name} must be a torch.Tensor, got {type(tensor)}"
            )

        if tensor.dim() not in [3, 4]:
            raise InvalidInputError(
                f"{name} must be 3D (HWC) or 4D (BHWC) tensor, got {tensor.dim()}D"
            )

        # Check shape
        if tensor.dim() == 4:
            batch_size, height, width, channels = tensor.shape
            if batch_size != 1:
                raise InvalidInputError(
                    f"{name} batch size must be 1, got {batch_size}"
                )
        else:
            height, width, channels = tensor.shape

        if height <= 0 or width <= 0:
            raise InvalidInputError(
                f"{name} dimensions must be positive, got {height}x{width}"
            )

        if channels not in [1, 3, 4]:
            raise InvalidInputError(
                f"{name} must have 1, 3, or 4 channels, got {channels}"
            )

        # Check value range (should be [0, 1] for ComfyUI)
        if tensor.min() < 0 or tensor.max() > 1:
            raise InvalidInputError(f"{name} values must be in [0, 1] range")

    @staticmethod
    def validate_numeric_input(
        value: Union[int, float],
        name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> None:
        """
        Validate numeric input parameters.

        Args:
            value: Input value to validate
            name: Name of the parameter for error messages
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)

        Raises:
            InvalidInputError: If value is invalid
        """
        if not isinstance(value, (int, float)):
            raise InvalidInputError(f"{name} must be a number, got {type(value)}")

        if min_val is not None and value < min_val:
            raise InvalidInputError(f"{name} must be >= {min_val}, got {value}")

        if max_val is not None and value > max_val:
            raise InvalidInputError(f"{name} must be <= {max_val}, got {value}")

    @staticmethod
    def validate_string_input(
        value: str, name: str, allowed_values: Optional[list] = None
    ) -> None:
        """
        Validate string input parameters.

        Args:
            value: Input value to validate
            name: Name of the parameter for error messages
            allowed_values: List of allowed string values

        Raises:
            InvalidInputError: If value is invalid
        """
        if not isinstance(value, str):
            raise InvalidInputError(f"{name} must be a string, got {type(value)}")

        if allowed_values is not None and value not in allowed_values:
            raise InvalidInputError(
                f"{name} must be one of {allowed_values}, got '{value}'"
            )

    @classmethod
    def create_input_types(cls, **kwargs) -> Dict[str, Any]:
        """
        Helper method to create INPUT_TYPES dictionary with validation.

        Args:
            **kwargs: Key-value pairs for input definitions

        Returns:
            INPUT_TYPES dictionary
        """
        return {"required": kwargs}

    def __init_subclass__(cls, **kwargs):
        """Ensure subclasses implement required abstract methods."""
        super().__init_subclass__(**kwargs)

        # Validate that abstract properties are defined
        required_attrs = ["RETURN_TYPES", "FUNCTION", "CATEGORY"]
        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise TypeError(f"Subclass {cls.__name__} must define {attr}")

        # Validate INPUT_TYPES is callable
        if not callable(getattr(cls, "INPUT_TYPES", None)):
            raise TypeError(
                f"Subclass {cls.__name__} must define INPUT_TYPES as a classmethod"
            )
