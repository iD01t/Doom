import pytest
import numpy as np
from PIL import Image, ImageChops

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sprite_forge_enhanced import ImageProcessor

@pytest.fixture
def processor():
    return ImageProcessor

@pytest.fixture
def sample_image():
    img = Image.new('RGBA', (10, 10), (0, 0, 0, 0))
    # Use a color that is not at the maximum value to test enhancement
    for x in range(2, 8):
        for y in range(2, 8):
            img.putpixel((x, y), (200, 50, 50, 255))
    return img

def test_enhance_sprite(processor, sample_image):
    original_pixel = sample_image.getpixel((4, 4))
    enhanced = processor.enhance_sprite(sample_image, brightness=1.2, contrast=1.2)
    enhanced_pixel = enhanced.getpixel((4, 4))
    assert enhanced_pixel[0] > original_pixel[0]
    assert enhanced_pixel[0] <= 255

def test_pixelate_image(processor, sample_image):
    pixelated = processor.pixelate_image(sample_image, factor=4)
    assert pixelated.size == sample_image.size
    color = pixelated.getpixel((4, 4))
    for x in range(4, 8):
        for y in range(4, 8):
            assert pixelated.getpixel((x, y)) == color

def test_apply_doom_palette(processor, sample_image):
    doomed = processor.apply_doom_palette(sample_image)
    assert doomed.size == sample_image.size
    assert doomed.getpixel((0, 0))[3] == 0
    assert doomed.getpixel((4, 4))[3] == 255
    assert doomed.getpixel((4, 4)) != sample_image.getpixel((4, 4))

def test_auto_crop(processor, sample_image):
    bordered = Image.new('RGBA', (20, 20), (0, 0, 0, 0))
    bordered.paste(sample_image, (5, 5))
    cropped = processor.auto_crop(bordered, threshold=10)
    assert cropped.size == (6, 6)
