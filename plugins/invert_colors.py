from sprite_forge_pro import BasePlugin, PluginInfo, Image, ImageOps

class InvertColorsPlugin(BasePlugin):
    info = PluginInfo(
        name="Invert Colors",
        version="1.0",
        author="Jules",
        description="Inverts the colors of all images in the current project.",
        category="Color"
    )

    def run(self, **kwargs):
        """
        Iterates through all sprites and frames, inverting the color of each image.
        Note: This is a destructive operation and is not currently undoable.
        """
        project = self.core.project
        if not project:
            print("No project loaded.")
            return

        for sprite_name, sprite in project.sprites.items():
            for frame_name, frame in sprite.frames.items():
                if frame.image:
                    # We need to handle both RGB and RGBA images for inversion
                    if frame.image.mode == 'RGBA':
                        r, g, b, a = frame.image.split()
                        rgb_image = Image.merge('RGB', (r, g, b))
                        inverted_rgb = ImageOps.invert(rgb_image)
                        r, g, b = inverted_rgb.split()
                        frame.image = Image.merge('RGBA', (r, g, b, a))
                    elif frame.image.mode == 'RGB':
                        frame.image = ImageOps.invert(frame.image)

                    print(f"Inverted colors for frame: {frame_name}")

        # In a real scenario, you would refresh the UI here
        print("Invert Colors plugin finished.")
