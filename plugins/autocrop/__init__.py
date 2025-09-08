def run(image, core_engine, threshold=10):
    """
    Wraps the core ImageProcessor's auto_crop functionality.
    """
    if not image or not core_engine:
        return None

    return core_engine.image_processor.auto_crop(image, transparency_threshold=threshold)
