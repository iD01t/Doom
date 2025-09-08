def run(image, core_engine, r=0, g=0, b=0, tolerance=20):
    """
    Wraps the core ImageProcessor's remove_background_fast functionality.
    """
    if not image or not core_engine:
        return None

    color_to_remove = (r, g, b)
    return core_engine.image_processor.remove_background_fast(image, color=color_to_remove, tolerance=tolerance)
