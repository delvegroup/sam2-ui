#!/usr/bin/env python3
"""
Gradio UI for SAM2 Object Extraction
Interactive interface for clicking on objects to extract them
"""

import cv2
from datetime import datetime
import gradio as gr
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global model cache
cached_model = None
cached_model_size = None

# Global state for click points
click_points = []
current_image = None

def get_model(model_size="large"):
    """Get or cache the SAM2 model"""
    global cached_model, cached_model_size
    
    if cached_model is None or cached_model_size != model_size:
        logger.info(f"Loading SAM2 model: {model_size}")
        cached_model = load_sam2_model(model_size)
        cached_model_size = model_size
    
    return cached_model


def load_sam2_model(model_size="large"):
    """Load SAM2 model from Hugging Face"""
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    model_id = f"facebook/sam2-hiera-{model_size}"
    
    logging.info(f"Loading model from Hugging Face: {model_id}")
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    
    return predictor

def add_point_to_image(image, points):
    """Draw points on the image"""
    if image is None:
        return image
    
    # Convert to numpy and ensure RGB
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Draw points
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(x), int(y)), 10, (255, 0, 0), 2)
        # Add point number
        cv2.putText(img, str(i+1), (int(x)+12, int(y)-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return img

def handle_click(evt: gr.SelectData):
    """Handle click on image"""
    global click_points, current_image
    
    # Add clicked point
    click_points.append((evt.index[0], evt.index[1]))
    
    # Update image with points
    img_with_points = add_point_to_image(current_image, click_points)
    
    status = f"Added point {len(click_points)}: ({evt.index[0]}, {evt.index[1]})"
    
    return img_with_points, status

def clear_points():
    """Clear all clicked points"""
    global click_points
    click_points = []
    return current_image, "Points cleared"

def segment_with_points(image, model_size="large"):
    """Segment image using clicked points"""
    global click_points
    
    if image is None:
        return None, "Please upload an image"
    
    if len(click_points) == 0:
        return None, "Please click on the object you want to extract"
    
    try:
        # Get model
        predictor = get_model(model_size)
        
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Ensure RGB
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Set image
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_np)
            
            # Convert click points to numpy arrays
            point_coords = np.array(click_points)
            point_labels = np.ones(len(click_points), dtype=np.int32)  # All foreground
            
            logger.info(f"Using {len(point_coords)} points for segmentation")
            
            # Predict masks
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        
        # Log mask info
        logger.info(f"Got {len(masks)} masks with scores: {scores}")
        
        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        # Create RGBA output with clean transparent areas
        h, w = image_np.shape[:2]
        rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Only copy RGB values where mask is True
        mask_bool = mask.astype(bool)
        rgba_image[:, :, :3][mask_bool] = image_np[mask_bool]
        rgba_image[:, :, 3] = mask.astype(np.uint8) * 255
        
        # Set transparent pixels to white (optional, but cleaner)
        rgba_image[:, :, :3][~mask_bool] = 255
        
        # Crop to mask bounds
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            # Add some padding
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w - 1, x_max + padding)
            y_max = min(h - 1, y_max + padding)
            
            cropped = rgba_image[y_min:y_max+1, x_min:x_max+1]
            result_image = Image.fromarray(cropped, 'RGBA')
            
            return result_image, f"Extracted object successfully! Score: {scores[best_idx]:.3f}"
        else:
            return None, "No object found at the clicked points"
            
    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        return None, f"Error: {str(e)}"

def save_result(image, filename):
    """Save the extracted result"""
    if image is None:
        return "No image to save"
    
    try:
        # Create output directory
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extracted_{timestamp}.png"
        
        # Ensure .png extension
        if not filename.endswith('.png'):
            filename += '.png'
        
        # Save image with proper PNG settings
        output_path = output_dir / filename
        # Ensure we save with proper alpha channel
        image.save(output_path, 'PNG', optimize=True)
        
        return f"Saved to: {output_path}"
    except Exception as e:
        return f"Error saving: {str(e)}"

def create_ui():
    """Create the Gradio interface"""
    with gr.Blocks(title="SAM2 Object Extraction") as demo:
        gr.Markdown("# SAM2 Object Extraction")
        gr.Markdown("Click on the object you want to extract. You can click multiple points for better accuracy.")
        
        with gr.Row():
            with gr.Column():
                # Model selection
                model_size = gr.Dropdown(
                    choices=["tiny", "small", "base", "large"],
                    value="large",
                    label="Model Size",
                    info="Larger models are more accurate but slower"
                )
                
                # Image input
                image_input = gr.Image(
                    label="Click on the object to extract",
                    type="numpy",
                    interactive=True
                )
                
                # Buttons
                with gr.Row():
                    clear_btn = gr.Button("Clear Points")
                    extract_btn = gr.Button("Extract Object", variant="primary")
                
            with gr.Column():
                # Output image
                output_image = gr.Image(
                    label="Extracted Object",
                    type="pil",
                    image_mode="RGBA"
                )
                
                # Status message
                status = gr.Textbox(label="Status", interactive=False)
                
                # Save controls
                with gr.Row():
                    filename_input = gr.Textbox(
                        label="Filename (optional)",
                        placeholder="extracted_object.png"
                    )
                    save_btn = gr.Button("Save Result")
                
                save_status = gr.Textbox(label="Save Status", interactive=False)
        
        # Examples
        gr.Markdown("### Tips")
        gr.Markdown("""
        - Click directly on the object you want to extract
        - For better results, click multiple points on the object
        - Avoid clicking on the background
        - The model will automatically find the object boundaries
        """)
        
        # Event handlers
        def update_current_image(img):
            """Update the current image when a new one is uploaded"""
            global current_image, click_points
            current_image = img
            click_points = []
            return img
        
        # Update current image when uploaded
        image_input.upload(
            fn=update_current_image,
            inputs=[image_input],
            outputs=[image_input]
        )
        
        # Handle clicks on image
        image_input.select(
            fn=handle_click,
            inputs=[],
            outputs=[image_input, status]
        )
        
        # Clear points button
        clear_btn.click(
            fn=clear_points,
            inputs=[],
            outputs=[image_input, status]
        )
        
        # Extract button
        extract_btn.click(
            fn=lambda img, size: segment_with_points(current_image, size),
            inputs=[image_input, model_size],
            outputs=[output_image, status]
        )
        
        save_btn.click(
            fn=save_result,
            inputs=[output_image, filename_input],
            outputs=[save_status]
        )
    
    return demo

def main():
    """Main entry point"""
    # Create and launch UI
    demo = create_ui()
    
    logger.info("Starting Gradio UI...")
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True for public URL
        inbrowser=True
    )

if __name__ == "__main__":
    main()