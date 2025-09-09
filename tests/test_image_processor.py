import pytest
import numpy as np
from PIL import Image, ImageChops

# Add the root directory to the Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sprite_forge_pro import ImageProcessor

@pytest.fixture
def processor():
    """Returns an instance of ImageProcessor."""
    return ImageProcessor()

@pytest.fixture
def sample_image():
    """Creates a 10x10 sample image with a red square on a transparent background."""
    img = Image.new('RGBA', (10, 10), (0, 0, 0, 0))
    for x in range(2, 8):
        for y in range(2, 8):
            img.putpixel((x, y), (255, 0, 0, 255))
    return img

def test_add_padding(processor, sample_image):
    """Test that padding increases image dimensions."""
    padded = processor.add_padding(sample_image, 2)
    assert padded.size == (14, 14)
    # Check that a corner pixel is transparent
    assert padded.getpixel((0, 0)) == (0, 0, 0, 0)
    # Check that a pixel from the original image is in the new position
    assert padded.getpixel((4, 4)) == (255, 0, 0, 255)

def test_scale_image(processor, sample_image):
    """Test image scaling."""
    scaled = processor.scale_image(sample_image, 2.0)
    assert scaled.size == (20, 20)

def test_adjust_brightness_contrast(processor):
    """Test brightness and contrast adjustment with a mid-gray image."""
    # Create a mid-gray image that is not at the extremes of the color range
    gray_image = Image.new('RGB', (10, 10), (128, 128, 128))

    adjusted = processor.adjust_brightness_contrast(gray_image, brightness=1.5, contrast=1.2)
    assert adjusted.size == gray_image.size

    # The image should be different from the original
    diff = ImageChops.difference(gray_image, adjusted)
    assert diff.getbbox() is not None

    # The new pixel should be brighter
    original_pixel = gray_image.getpixel((5, 5))
    adjusted_pixel = adjusted.getpixel((5, 5))
    assert adjusted_pixel[0] > original_pixel[0]

def test_convert_to_grayscale(processor, sample_image):
    """Test grayscale conversion."""
    gray = processor.convert_to_grayscale(sample_image)
    assert gray.size == sample_image.size
    pixel = gray.getpixel((4, 4))
    # For a red pixel (255,0,0), the grayscale value with default weights should be around 76.
    # The pixel will be (R,G,B,A)
    assert pixel[0] == pixel[1] == pixel[2]
    assert abs(pixel[0] - 76) < 2 # Allow for small rounding differences

def test_replace_color(processor, sample_image):
    """Test color replacement."""
    blue = (0, 0, 255, 255)
    replaced = processor.replace_color(sample_image, find_color=(255, 0, 0), replace_color=blue[:3])
    assert replaced.getpixel((4, 4)) == blue

def test_create_outline(processor, sample_image):
    """Test outline creation."""
    outlined = processor.create_glow_or_outline(sample_image, color=(0, 255, 0), size=1, mode='outline')
    assert outlined.size == sample_image.size
    # Check a pixel in the original red square is still red
    assert outlined.getpixel((4, 4)) == (255, 0, 0, 255)
    # Check a pixel in the new outline is green
    assert outlined.getpixel((1, 4)) == (0, 255, 0, 255)
