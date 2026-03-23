import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

LAST_CONV_LAYER = "out_relu"  # MobileNetV2 last conv layer

def get_gradcam_heatmap(model, img_array):
    """
    Generates Grad-CAM heatmap for the predicted class.
    Returns heatmap as numpy array (0-1 range).
    """
    # Build grad model up to last conv layer
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index   = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads       = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(original_img_array, heatmap, alpha=0.4):
    """
    Overlays Grad-CAM heatmap onto the original image.
    Returns superimposed image as numpy array (0-1 range).
    """
    # Resize heatmap to image size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(
        (original_img_array.shape[1], original_img_array.shape[0]),
        Image.LANCZOS
    )
    heatmap_resized = np.array(heatmap_img) / 255.0

    # Apply colormap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]

    # Blend with original
    superimposed = heatmap_colored * alpha + original_img_array * (1 - alpha)
    return np.clip(superimposed, 0, 1)