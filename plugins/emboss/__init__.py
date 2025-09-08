from PIL import ImageFilter

def run(image):
    """
    Applies an emboss effect to the image.
    """
    if image:
        return image.filter(ImageFilter.EMBOSS)
    return None
