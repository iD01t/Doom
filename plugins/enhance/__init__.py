def run(image, core_engine, enhancement_type="brightness", factor=1.5):
    """
    Uses the core ImageProcessor to apply enhancements.
    This demonstrates a plugin's ability to interact with the main application.
    """
    if not image or not core_engine:
        return None

    # Access the core image processor via the core_engine
    image_processor = core_engine.image_processor

    if hasattr(image_processor, 'adjust_enhancement'):
        return image_processor.adjust_enhancement(image, enhancement_type, factor)

    return image
