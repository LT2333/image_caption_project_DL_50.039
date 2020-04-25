import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
import os
import argparse

import sample
from sample import Vocabulary

kivy.require('1.10.1')
Window.size = (1000,600)



class MainScreen(BoxLayout):

    def selected(self, filename):
        self.ids.img.source = filename[0]

        sentences = sample.start_app_pred(self.ids.img.source, args.encoder_path, args.decoder_path)
        self.ids.pred.text = "Prediction: "+sentences[8:-5].capitalize()


class ImageCaptionApp(App):
    kv_directory = 'kivy_utils'
    def build(self):
        return MainScreen()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
    args = parser.parse_args()
    ImageCaptionApp().run()
