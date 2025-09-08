from PIL import ImageFilter

def run(image, radius=2.0):
    """
    Applies a Gaussian blur to the image.
    This function is the entry point for the plugin.
    """
    if image:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    return None
