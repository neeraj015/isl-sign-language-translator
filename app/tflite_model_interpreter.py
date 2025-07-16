import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="models/best_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image):
    image = image.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], [image])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output
