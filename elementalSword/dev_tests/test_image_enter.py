from kivy.app import App
from kivy.uix.image import Image
from kivy.core.window import Window
from PIL import Image as PILImage

class ImageHoverApp(App):
    def build(self):
        self.img = Image(source='images/cities/tamarania.png', allow_stretch=True)
        self.img.fit_mode = 'contain'
        self.pil_img = PILImage.open('images/cities/region/common_regions.png')
        Window.bind(mouse_pos=self.on_mouse_pos)
        return self.img

    def on_mouse_pos(self, window, pos):
        # Check if the mouse is over the widget
        if not self.img.collide_point(*pos):
            return

        # Get the position and size of the widget
        wx, wy = self.img.pos
        w_width, w_height = self.img.size

        # Calculate the scale factor and offset based on 'contain' mode
        img_ratio = self.pil_img.width / self.pil_img.height
        widget_ratio = w_width / w_height
        if img_ratio > widget_ratio:
            # Image is wider than the widget
            scale = w_width / self.pil_img.width
            offset_x = 0
            offset_y = (w_height - (self.pil_img.height * scale)) / 2
        else:
            # Image is taller than the widget
            scale = w_height / self.pil_img.height
            offset_x = (w_width - (self.pil_img.width * scale)) / 2
            offset_y = 0

        # Adjust mouse coordinates to image coordinates
        mx, my = pos[0] - wx - offset_x, pos[1] - wy - offset_y
        nx, ny = int(mx / scale), int(my / scale)

        # Adjust for the y coordinate being inverted in PIL
        ny = self.pil_img.height - ny - 1

        if 0 <= nx < self.pil_img.width and 0 <= ny < self.pil_img.height:
            # Get the color of the pixel
            pixel = self.pil_img.getpixel((nx, ny))
            hex_value = '#%02x%02x%02x' % pixel[:3]
            print(f"Mouse over pixel at ({nx}, {ny}), Color: {pixel}, Hex: {hex_value}")

if __name__ == '__main__':
    ImageHoverApp().run()

# class SlcImage(Image, HoverBehavior):
#     def __init__(self, img, x, y, **kwargs):
#         super(SlcImage, self).__init__(**kwargs)
#         self.source = f'images\\cities\\selection\\{img}.png'
#         Window.bind(mouse_pos = self.on_mouse_move)
#     def on_mouse_move(self, *args):
