from PIL import Image, ImageDraw, ImageFont
import numpy as np

def quantize_image(image, bit_depth):
    """Quantize an image to the specified bit depth per channel."""
    if bit_depth < 1:
        raise ValueError("Bit depth must be at least 1.")
    levels = 2 ** bit_depth
    scale_factor = 255 / (levels - 1)  # Scale up to 255 for proper brightness
    quantized = image.point(lambda p: round(p / scale_factor) * scale_factor)
    return quantized

def create_color_gradient(width, height):
    """Create a red-to-blue horizontal gradient."""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        t = x / (width - 1)  # Normalize to [0, 1]
        gradient[:, x, 0] = int(255 * (1 - t))  # Red decreases
        gradient[:, x, 1] = int(abs(64 - t))  # Red decreases
        gradient[:, x, 2] = int(255 * t)        # Blue increases
    gradient_image = Image.fromarray(gradient, 'RGB')
    return gradient_image

def create_quantization_demo(input_image_path, output_image_path):
    """Create a demonstration of quantization with 32, 16, 8, 4, and 2-bit sections."""
    original = Image.open(input_image_path).convert("RGB")
    width, height = original.size
    gradient_height = int(height * 0.05)  # 5% of image height

    # Define bit depths and calculate section width
    bit_depths = [32, 16, 8, 4]
    section_width = width // len(bit_depths)

    # Create output image
    output_image = Image.new("RGB", (width, height))

    # Create and apply quantization for each section
    for i, bit_depth in enumerate(bit_depths):
        if bit_depth == 32:
            quantized_section = original
        else:
            quantized_section = quantize_image(original, bit_depth // 3)

        # Crop and paste the quantized section
        section = quantized_section.crop((i * section_width, 0, (i + 1) * section_width, height))
        output_image.paste(section, (i * section_width, 0))

        # Add gradient on top
        gradient = create_color_gradient(section_width, gradient_height)
        quantized_gradient = quantize_image(gradient, bit_depth // 3) if bit_depth < 32 else gradient
        output_image.paste(quantized_gradient, (i * section_width, 0))

    # Add labels for each section
    draw = ImageDraw.Draw(output_image)
    label_height = 20

    # Specify font and size
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust to your font path
    font_size = 24  # Increase font size here
    font = ImageFont.truetype(font_path, font_size)

    for i, bit_depth in enumerate(bit_depths):
        label = f"{bit_depth}-bit"
        text_x = i * section_width + 10
        text_y = height - label_height - font_size
        draw.text((text_x, text_y), label, fill="white", font=font)

    # Save the output image in WebP format
    output_image.save(output_image_path, format="WEBP")
    print(f"Quantization demo saved to {output_image_path}")
    
    
# Example usage
input_image_path = "../imgs/finetuning.webp"  # Replace with your input WebP image path
output_image_path = "../imgs/finetuning_quantization_demo.webp"
create_quantization_demo(input_image_path, output_image_path)
