import cv2
import tensorflow as tf
import tensorflow_hub as hub

from supported_models import SupportedModels


class Model():

    def __init__(self, model_name):
        if model_name == SupportedModels.movenet_lightning:
            
            self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.input_size = 192
            self.model = self.module.signatures['serving_default']

        elif model_name == SupportedModels.movenet_thunder:
            self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            self.input_size = 256
            self.model = self.module.signatures['serving_default']

        else:
            raise ValueError("Unsupported model name in Model class: %s" % model_name)
        
        self.model_name = model_name
        self.cap = cv2.VideoCapture(0)
    
    def run(self):
        if self.model_name == SupportedModels.movenet_lightning:
            return self.movenet_lightning()
        elif self.model_name == SupportedModels.movenet_thunder:
            return self.movenet_thunder()
    
    def movenet_lightning(self):
        if self.cap.isOpened():

            # Get a frame from the cap source
            ret, frame = self.cap.read()

            if ret is True:
        
                # Mirror the frame
                frame = cv2.flip(frame, 2)

                # Resize the image for detection
                img = frame.copy()
                input_img = tf.expand_dims(img, axis=0)
                input_img = tf.image.resize_with_pad(input_img, self.input_size, self.input_size)
                input_img = tf.cast(input_img, dtype=tf.int32)

                # Pose estimation model inference
                outputs = self.model(input_img)
                return outputs['output_0'].numpy().tolist()[0]
    
    def movenet_thunder(self):
        if self.cap.isOpened():

            # Get a frame from the cap source
            ret, frame = self.cap.read()

            if ret is True:
        
                # Mirror the frame
                frame = cv2.flip(frame, 2)

                # Resize the image for detection
                img = frame.copy()
                input_img = tf.expand_dims(img, axis=0)
                input_img = tf.image.resize_with_pad(input_img, self.input_size, self.input_size)
                input_img = tf.cast(input_img, dtype=tf.int32)

                # Pose estimation model inference
                outputs = self.model(input_img)
                return outputs['output_0'].numpy().tolist()[0]
