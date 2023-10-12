# import the necessary packages
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import os
import cv2
import sys
import numpy as np
from imutils import paths

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = 'global_average_pooling2d_GlobalAveragePooling2D1'
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
                self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
                # check to see if the layer has a 4D output
                if len(layer.output_shape) == 4:
                        return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer('sequential_1').get_layer('model1').get_layer('Conv_1').output,
                        self.model.output])
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap


    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)

dir_path = sys.argv[2]
img_paths = list(paths.list_images(dir_path))

model = load_model(sys.argv[1])
labels = ['healthy', 'partially injured', 'fully_ruptured']

print(model.get_layer('sequential_1').get_layer('model1').summary())
print([layer.name for layer in model.get_layer('sequential_1').get_layer('model1').layers])

for img_path in img_paths:
    print(img_path)

    img = cv2.imread(img_path)
    h, w, c = img.shape

    crop_size = min(h, w)
    c_x = w // 2
    c_y = h // 2
    
    xmin = c_x - crop_size // 2
    xmax = c_x + crop_size // 2
    ymin = c_y - crop_size // 2
    ymax = c_y + crop_size // 2
    
    center_crop = img[ymin:ymax, xmin:xmax, :] 
    
    img_rgb = cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))

    img_rgb = np.asarray(img_rgb, dtype=np.float32).reshape(1, 224, 224, 3)
    img_rgb = (img_rgb / 127.5) - 1

    pred = model.predict(img_rgb)[0]

    idx = np.argmax(pred)

    label = labels[idx]
    print(label, ' ', pred[idx])

    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, idx)
    heatmap = cam.compute_heatmap(img_rgb)
    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, img, alpha=0.5)


    # draw the predicted label on the output image
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2)
    # display the original image and resulting heatmap and output image
    # to our screen
    cv2.imwrite(sys.argv[3] + os.path.sep + os.path.basename(img_path), output)
