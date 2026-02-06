import os
import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
import subprocess # Digunakan untuk handling error yang lebih baik
import shutil

# 1. Config and preprocess
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3 
NUM_CLASSES = 1
DEEP_SUPERVISION = True 

# 2. Metrics and loss functions
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1e-5 
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# 3. Advanced building blocks
def conv_block(input_tensor, num_filters, dropout_rate=0.0):
    x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if dropout_rate > 0.0:
        x = layers.SpatialDropout2D(dropout_rate)(x)

    x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

# 4. U-Net++ Architecture
def build_unet_plus_plus(input_shape, deep_supervision=False):
    inputs = layers.Input(input_shape)
    nb_filter = [32, 64, 128, 256, 512] 

    # Encoder
    x_0_0 = conv_block(inputs, nb_filter[0], dropout_rate=0.0)
    p_0 = layers.MaxPooling2D((2, 2))(x_0_0)

    x_1_0 = conv_block(p_0, nb_filter[1], dropout_rate=0.1)
    p_1 = layers.MaxPooling2D((2, 2))(x_1_0)

    x_2_0 = conv_block(p_1, nb_filter[2], dropout_rate=0.2)
    p_2 = layers.MaxPooling2D((2, 2))(x_2_0)

    x_3_0 = conv_block(p_2, nb_filter[3], dropout_rate=0.3)
    p_3 = layers.MaxPooling2D((2, 2))(x_3_0)
    
    x_4_0 = conv_block(p_3, nb_filter[4], dropout_rate=0.4)

    # Decoder
    x_0_1 = conv_block(layers.concatenate([layers.UpSampling2D()(x_1_0), x_0_0]), nb_filter[0])
    x_1_1 = conv_block(layers.concatenate([layers.UpSampling2D()(x_2_0), x_1_0]), nb_filter[1])
    x_2_1 = conv_block(layers.concatenate([layers.UpSampling2D()(x_3_0), x_2_0]), nb_filter[2])
    x_3_1 = conv_block(layers.concatenate([layers.UpSampling2D()(x_4_0), x_3_0]), nb_filter[3])

    x_0_2 = conv_block(layers.concatenate([layers.UpSampling2D()(x_1_1), x_0_0, x_0_1]), nb_filter[0])
    x_1_2 = conv_block(layers.concatenate([layers.UpSampling2D()(x_2_1), x_1_0, x_1_1]), nb_filter[1])
    x_2_2 = conv_block(layers.concatenate([layers.UpSampling2D()(x_3_1), x_2_0, x_2_1]), nb_filter[2])

    x_0_3 = conv_block(layers.concatenate([layers.UpSampling2D()(x_1_2), x_0_0, x_0_1, x_0_2]), nb_filter[0])
    x_1_3 = conv_block(layers.concatenate([layers.UpSampling2D()(x_2_2), x_1_0, x_1_1, x_1_2]), nb_filter[1])

    x_0_4 = conv_block(layers.concatenate([layers.UpSampling2D()(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3]), nb_filter[0])

    if deep_supervision:
        out_1 = layers.Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', name='output_1')(x_0_1)
        out_2 = layers.Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', name='output_2')(x_0_2)
        out_3 = layers.Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', name='output_3')(x_0_3)
        out_4 = layers.Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', name='output_4')(x_0_4)
        model = models.Model(inputs=[inputs], outputs=[out_1, out_2, out_3, out_4], name="UNet_PlusPlus_DeepSup")
    else:
        output = layers.Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', name='final_output')(x_0_4)
        model = models.Model(inputs=[inputs], outputs=[output], name="UNet_PlusPlus_Standard")

    return model

# 5. Compilation
def get_compiled_model():
    model = build_unet_plus_plus((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), deep_supervision=DEEP_SUPERVISION)
    
    if DEEP_SUPERVISION:
        losses = {'output_1': dice_loss, 'output_2': dice_loss, 'output_3': dice_loss, 'output_4': dice_loss}
        loss_weights = {'output_1': 1.0, 'output_2': 1.0, 'output_3': 1.0, 'output_4': 1.0}
        model.compile(optimizer=Adam(learning_rate=1e-4), loss=losses, loss_weights=loss_weights, metrics=[dice_coef])
    else:
        model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=['accuracy', dice_coef])
    
    return model

# 6. Main execution
if __name__ == "__main__":
    rbc_model = get_compiled_model()
    print(f"\n[INFO] U-Net++ Built. Deep Supervision: {DEEP_SUPERVISION}")

    print("\n[INFO] Starting Standard ONNX Export...")
    
    temp_model_dir = "temp_rbc_model"
    if os.path.exists(temp_model_dir):
        shutil.rmtree(temp_model_dir)
    
    # Save using tf.saved_model
    try:
        tf.saved_model.save(rbc_model, temp_model_dir)
    except Exception as e:
        print(f"[WARN] Standard save failed, trying export: {e}")
        rbc_model.export(temp_model_dir)

    output_path = "unet_plusplus_rbc.onnx"

    # Command construction
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", temp_model_dir,
        "--output", output_path,
        "--opset", "12"
    ]
    
    print(f"[CMD] Executing: {' '.join(cmd)}")
    
    # Use subprocess.check_call and if errors, stop script then show it
    try:
        subprocess.check_call(cmd)
        print(f"\n[SUCCESS] Model exported to '{output_path}' (Standard Layout).")
        print("[INFO] Please COPY this new .onnx file to your C++ build folder and overwrite the old one!")
    except subprocess.CalledProcessError:
        print(f"\n[FATAL ERROR] TF2ONNX conversion FAILED.")
        print("[TIP] Ensure you have installed numpy<2.0: pip install \"numpy<2.0\"")
    
    # Clean up
    if os.path.exists(temp_model_dir):
        shutil.rmtree(temp_model_dir)
