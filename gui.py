import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
import argparse
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

import sample
from sample import Vocabulary

kivy.require('1.10.1')
Window.size = (1000,600)

class MainScreen(BoxLayout, FloatLayout):
    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select Image", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        self.ids.img.source = filename[0]

        sentences = sample.start_app_pred(self.ids.img.source, args.encoder_path, args.decoder_path)
        self.ids.pred.text = "Prediction: "+sentences[8:-5].capitalize()


class LoadDialog(BoxLayout, FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    def selected(self, filename):
        ImgPred().selected(filename)
        


class ImgPred(BoxLayout, FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def selected(self, filename):

        sentences = sample.start_app_pred(filename[0], args.encoder_path, args.decoder_path)

        box = BoxLayout(orientation='vertical')
        box.add_widget(Image(source=filename[0]))
        btn = Button(text='Click to Select Another Image', size_hint=(1, 0.2), font_size=20)
        btn.bind(on_press=self.show_load)
        box.add_widget(btn)

        self._popup = Popup(title="Prediction: "+sentences[8:-5].capitalize(), title_size=(25), title_align='center',
                            content=box, size_hint=(0.9, 0.9))
        self._popup.open()



    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self, *args):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select Image", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()



class ImageCaptionApp(App):
    kv_directory = 'kivy_utils'
    def build(self):
        return MainScreen()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/my-encoder-5-3000-t4-resnext.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/my-decoder-5-3000-t4-resnext.ckpt', help='path for trained decoder')
    args = parser.parse_args()
    ImageCaptionApp().run()
