# ComfyUI Custom Image Editing Nodes

This package provides custom nodes for ComfyUI to perform image scaling and compositing.

## Installation

1. Clone or download this repository into your ComfyUI's `custom_nodes` directory.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Restart ComfyUI.

## Nodes

### Scale Image Node
- **Inputs:**
  - `image`: The input image to scale.
  - `target_dimension`: The target width or height value.
  - `dimension_type`: Choose "width" or "height" to specify which dimension to scale.
- **Output:** Scaled image maintaining aspect ratio.

### Composite Image Node
- **Inputs:**
  - `source_image`: The image to composite onto the destination.
  - `destination_image`: The base image.
  - `x_position`: Relative x position (0.0 = left edge, 1.0 = right edge).
  - `y_position`: Relative y position (0.0 = top edge, 1.0 = bottom edge).
  - `alignment`: Alignment mode - "top_left" (default) or "center".
- **Output:** Composited image.

## Usage
Load the nodes in ComfyUI and connect them in your workflow as needed.