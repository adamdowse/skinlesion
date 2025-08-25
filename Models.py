import tensorflow as tf
from tensorflow.keras import layers, Model, Layer

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Conv2DTranspose,
    UpSampling2D,
    Concatenate,
    BatchNormalization,
    ReLU
)
import matplotlib.pyplot as plt
import cv2
import numpy as np


class Res2NetBlock(Layer):
    def __init__(self, in_depth, out_depth, stride=1, downsample=None, basewidth=26, scale=4, stype="normal"):
        """
        Res2Net block implementation.
        Args:
            filters (int): Number of filters for convolution layers.
            n (int): Number of splits for feature maps.
        """
        super(Res2NetBlock, self).__init__()
        self.n = scale

        self.initial_conv = layers.Conv2D(in_depth, 1, activation='relu', padding='same') 
        self.split_convs = [layers.Conv2D(in_depth // self.n, 3, activation='relu', padding='same') for _ in range(self.n)]
        self.concat = layers.Concatenate(axis=-1)
        self.final_conv = layers.Conv2D(out_depth, 1, activation='relu', padding='same')

    def call(self, inputs):
        """
        Forward pass for Res2Net block.
        Args:
            inputs (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor after Res2Net block.
        """
        # Split input into n parts
        #tf.print("Input shape before initial conv:", inputs.shape)
        inputs = self.initial_conv(inputs)
        #tf.print("Input shape after initial conv:", inputs.shape)
        splits = tf.split(inputs, self.n, axis=-1)
        #tf.print("Number of splits:", len(splits))
        #tf.print("Shape of each split:", [split.shape for split in splits])
        outputs = []
        
        # Process each split with its corresponding convolution
        for i, conv in enumerate(self.split_convs):
            if i == 0:
                o = conv(splits[i])
                outputs.append(o)
            else:
                outputs.append(conv(splits[i] + outputs[-1]))

        
        # Concatenate outputs and apply final 1x1 convolution
        concatenated = self.concat(outputs)
        #tf.print("Shape after concatenation:", concatenated.shape)
        return self.final_conv(concatenated)
    
class SEBlock(Layer):
    def __init__(self, filters, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(filters // reduction, activation='relu')
        self.fc2 = layers.Dense(filters, activation='sigmoid')

    def call(self, inputs):
        #tf.print("Input shape in SEBlock:", inputs.shape)
        x = self.global_avg_pool(inputs)
        #tf.print("Shape after global average pooling:", x.shape)
        x = self.fc1(x)
        #tf.print("Shape after first dense layer:", x.shape)
        x = self.fc2(x)
        #tf.print("Shape after second dense layer:", x.shape)
        o = inputs * tf.reshape(x, [-1, 1, 1, inputs.shape[-1]])
        #tf.print("Output shape after SEBlock:", o.shape)
        return o

class PyramidDilatedConv(Layer):
    def __init__(self, filters, output_filters, dilation_increase=2, max_num_dilations=4):
        dilation_rates = [dilation_increase ** i for i in range(max_num_dilations)]
        super(PyramidDilatedConv, self).__init__()
        self.convs = [layers.Conv2D(filters, 3, padding='same', dilation_rate=dilation) for dilation in dilation_rates]
        self.concat = layers.Concatenate(axis=-1)
        self.final_conv = layers.Conv2D(output_filters, 1, activation='relu', padding='same')

    def call(self, inputs):
        outputs = []
        for i in range(len(self.convs)):
            branch = []
            for j in range(i + 1):
                if j == 0:
                    branch.append(self.convs[i - j](inputs))
                else:
                    branch.append(self.convs[i - j](tf.reduce_sum(branch, axis=0)))
                #tf.print(f"Branch {i} output shape:", branch[-1].shape)
            outputs.append(branch[-1])
        output = self.concat(outputs)
        #tf.print("Shape after concatenation in PyramidDilatedConv:", output.shape)
        return self.final_conv(output)


class ImageScaleLayer(Layer):
    def __init__(self, scale_factor):
        super(ImageScaleLayer, self).__init__()
        self.scale_factor = scale_factor
    def call(self, inputs):
        #tf.print("Input shape in ImageScaleLayer:", inputs.shape)
        s1 = tf.cast(tf.shape(inputs)[1], tf.float32)
        s2 = tf.cast(tf.shape(inputs)[2], tf.float32)
        output = tf.image.resize(inputs, (
            tf.cast(s1 * self.scale_factor, tf.int32),
            tf.cast(s2 * self.scale_factor, tf.int32)
        ),method='bilinear')
        return output

class Res2SEBlock(Layer):
    def __init__(self, filters, n):
        super(Res2SEBlock, self).__init__()
        self.filters = filters
        self.n = n
        self.R2N = Res2NetBlock(filters, filters, scale=n)
        self.SEA = SEBlock(filters, reduction=4)
        self.dropout = layers.Dropout(0.5)
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.ReLU()

    def call(self, inputs):
        #tf.print("Input shape in Res2SEBlock:", inputs.shape)
        x = self.R2N(inputs)
        #tf.print("Shape after Res2NetBlock:", x.shape)
        x = self.SEA(x)
        #tf.print("Shape after SEBlock:", x.shape)
        x = self.dropout(x)
        #tf.print("Shape after dropout:", x.shape)
        x = self.batch_norm(x)
        #tf.print("Shape after batch normalization:", x.shape)
        x = self.activation(x)
        #tf.print("Shape after activation:", x.shape)
        x = inputs + x  # Residual connection
        #tf.print("Shape after adding residual connection:", x.shape)
        return x

class MRPUEncoder(Layer):
    def __init__(self, img_shape=(64,64,3)):
        super(MRPUEncoder, self).__init__()
        self.img_shape = img_shape
        self.deduce_image_2 = ImageScaleLayer(scale_factor=0.5)
        self.deduce_image_4 = ImageScaleLayer(scale_factor=0.25)
        self.deduce_image_8 = ImageScaleLayer(scale_factor=0.125)

        self.conv1_64 = layers.Conv2D(64, 1, activation='relu', padding='same')
        self.conv1_128 = layers.Conv2D(128, 1, activation='relu', padding='same')
        self.conv1_256 = layers.Conv2D(256, 1, activation='relu', padding='same')

        self.res2se64 = Res2SEBlock(64, n=4)
        self.res2se128 = Res2SEBlock(128, n=4)
        self.res2se256 = Res2SEBlock(256, n=4)
        self.res2se512 = Res2SEBlock(512, n=4)

        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same') #TODO check strides
        self.concat = layers.Concatenate(axis=-1)

    def call(self, inputs):
        inp_1 = inputs
        inp_2 = self.deduce_image_2(inputs)
        inp_4 = self.deduce_image_4(inputs)
        inp_8 = self.deduce_image_8(inputs)

        o1 = self.conv1_64(inp_1)
        o2 = self.conv1_64(inp_2)
        o3 = self.conv1_128(inp_4)
        o4 = self.conv1_256(inp_8)

        o1 = self.res2se64(o1)
        o2 = self.res2se64(o2)
        o3 = self.res2se128(o3)
        o4 = self.res2se256(o4)

        ro1 = self.maxpool(o1)
        o2 = self.concat([o2, ro1])
        o2 = self.res2se128(o2)

        ro2 = self.maxpool(o2)
        o3 = self.concat([o3, ro2])
        o3 = self.res2se256(o3)

        ro3 = self.maxpool(o3)
        o4 = self.concat([o4, ro3])
        o4 = self.res2se512(o4)

        o5 = self.maxpool(o4)
        return o1, o2, o3, o4, o5

class MRPUDecoder(Layer):
    def __init__(self, num_classes):
        super(MRPUDecoder, self).__init__()
        self.num_classes = num_classes
        self.conv3_512 = layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv3_256 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv3_128 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv3_64 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv3_64_2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.final_conv = layers.Conv2D(num_classes, 1, activation='sigmoid', padding='same')
        self.upsample2 = ImageScaleLayer(scale_factor=2.0)
        self.concat = layers.Concatenate(axis=-1)

    def call(self, inputs):
        o1, o2, o3, o4, o5 = inputs

        eo5 = self.upsample2(o5)
        o4 = self.concat([o4, eo5])
        #print("o4 shape:", o4.shape)
        o4 = self.conv3_512(o4)
        o4 = self.conv3_256(o4)

        eo4 = self.upsample2(o4)
        o3 = self.concat([o3, eo4])
        #print("o3 shape:", o3.shape)
        o3 = self.conv3_256(o3)
        o3 = self.conv3_128(o3)

        eo3 = self.upsample2(o3)
        o2 = self.concat([o2, eo3])
        #print("o2 shape:", o2.shape)
        o2 = self.conv3_128(o2)
        o2 = self.conv3_64(o2)

        eo2 = self.upsample2(o2)
        o1 = self.concat([o1, eo2])
        #print("o1 shape:", o1.shape)
        o1 = self.conv3_64(o1)
        #print("o1 shape:", o1.shape)
        o1 = self.conv3_64_2(o1)
        #print("o1 shape:", o1.shape)
        o1 = self.final_conv(o1)
        #print("o1 shape:", o1.shape)
        return o1

class HairRemoval(Layer):
    """
    TensorFlow Layer for hair removal and inpainting using adaptive principle curvature (APC).
    For demonstration, inpainting is approximated with Gaussian blur.
    """
    def __init__(self, kernel_size=5, threshold=0.2, **kwargs):
        super(HairRemoval, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.threshold = threshold

    def build(self, input_shape):
        def gaussian_kernel(size, sigma=1.0):
            x = tf.range(-size // 2 + 1, size // 2 + 1)
            x = tf.cast(x, tf.float32)
            g = tf.exp(-(x**2) / (2.0 * sigma**2))
            g = g / tf.reduce_sum(g)
            return g

        gk = gaussian_kernel(self.kernel_size, sigma=2.0)
        gk2d = tf.tensordot(gk, gk, axes=0)
        gk2d = gk2d[:, :, tf.newaxis, tf.newaxis]
        input_channels = input_shape[-1]
        self.gaussian_kernel = tf.tile(
            tf.constant(gk2d, dtype=tf.float32),
            [1, 1, input_channels, 1]
        )

    def call(self, inputs):
        # Convert to grayscale for hair detection
        gray = tf.image.rgb_to_grayscale(inputs)
        # Use Sobel filter to detect edges (hairs)
        sobel = tf.image.sobel_edges(gray)
        edge_mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
        edge_mag = tf.squeeze(edge_mag, axis=-1)
        # Threshold to create hair mask
        hair_mask = tf.cast(edge_mag > self.threshold, tf.float32)
        hair_mask = tf.expand_dims(hair_mask, axis=-1)
        # Inpaint: blur the image and use blurred values where hair is detected
        blurred = tf.nn.depthwise_conv2d(inputs, self.gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')
        # Composite: use blurred where hair, original elsewhere
        output = inputs * (1 - hair_mask) + blurred * hair_mask
        return output

class MRPUNet(Model):
    def __init__(self, num_classes):
        super(MRPUNet, self).__init__()
        self.encoder = MRPUEncoder(img_shape=(576, 576, 3))
        self.decoder = MRPUDecoder(num_classes=num_classes)
        self.PDC = PyramidDilatedConv(512, 512, dilation_increase=2, max_num_dilations=4)
        self.res2se_512 = Res2SEBlock(512, n=4)

    def call(self, inputs):
        x = self.encoder(inputs)
        # Apply PyramidDilatedConv to the last output of the encoder
        x = list(x)
        #print("Encoder outputs:", [o.shape for o in x])
        x[-1] = self.res2se_512(x[-1])  # Assuming
        x[-1] = self.PDC(x[-1])  # Assuming x[-1] is the last output from the encoder
        x[-1] = self.res2se_512(x[-1])  # Apply Res2SEBlock to the last output
        x = self.decoder(x)
        return x
    def build(self, input_shape):
        # Build the model with the specified input shape
        inputs = tf.keras.Input(shape=input_shape[1:])
        self.call(inputs)
        super(MRPUNet, self).build(input_shape)

@tf.py_function(Tout=tf.float32)
def remove_hairs_apc_like(tf_image):
    # Convert TensorFlow tensor to numpy array
    image = (tf_image.numpy() * 255).astype(np.uint8)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blackhat morphological operation to find dark lines (hairs)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Threshold to create mask
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Inpaint using Telea's method (or cv2.INPAINT_NS for Navier-Stokes)
    inpainted_item = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # Convert inpainted image back to float32 for TensorFlow
    inpainted = tf.convert_to_tensor(inpainted_item, dtype=tf.float32) / 255.0 # Normalize to [0, 1]
    return inpainted


class ClassificationDecoder(Layer):
    """
    A decoder for image classification from the latent space.
    """
    def __init__(self, num_classes, name='classification_decoder', **kwargs):
        super(ClassificationDecoder, self).__init__(name=name, **kwargs)
        self.gap_layers = [GlobalAveragePooling2D() for _ in range(5)]
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(512, activation='relu')
        self.dense3 = Dense(256, activation='relu')
        self.dense4 = Dense(128, activation='relu')
        # The final output layer with softmax for classification
        self.classification_head = Dense(num_classes, activation='softmax', name='classification_output')

    def call(self, latent_space):
        #flatten each input
        x1,x2,x3,x4,x5 = [self.gap_layers[i](latent_space[i]) for i in range(5)]
        #concatenate the inputs
        x = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        #reduce the dimensionality to 256
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        #apply the classification head
        return self.classification_head(x)
    

# --- Segmentation Decoder ---
class SegmentationDecoder(Layer):
    """
    A decoder to generate segmentation maps from the latent space.
    This uses transposed convolutions to upsample the image back to its original size.
    """
    def __init__(self, num_seg_classes, name='segmentation_decoder', **kwargs):
        super(SegmentationDecoder, self).__init__(name=name, **kwargs)
        self.upconv1 = Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu') # 72x72 -> 144x144
        self.bn1 = BatchNormalization()
        
        self.upconv2 = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu') # 144x144 -> 288x288
        self.bn2 = BatchNormalization()
        
        self.upconv3 = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')  # 288x288 -> 576x576
        self.bn3 = BatchNormalization()

        # Final convolution to get the segmentation map
        self.segmentation_head = Conv2D(num_seg_classes, 1, activation='softmax', name='segmentation_output')

    def call(self, latent_space):
        x = self.upconv1(latent_space)
        x = self.bn1(x)
        x = self.upconv2(x)
        x = self.bn2(x)
        x = self.upconv3(x)
        x = self.bn3(x)
        return self.segmentation_head(x)

# --- 3. Combine into a single tf.keras.Model ---
class MRPUNetMT(Model):
    def __init__(self, num_seg_classes, num_classes):
        super(MRPUNetMT, self).__init__()
        self.encoder = MRPUEncoder(img_shape=(576, 576, 3))
        self.class_decoder = ClassificationDecoder(num_classes=num_classes)
        self.seg_decoder = MRPUDecoder(num_classes=num_seg_classes)
        #self.PDC = PyramidDilatedConv(512, 512, dilation_increase=2, max_num_dilations=4)
        self.res2se_512 = Res2SEBlock(512, n=4)

    def call(self, inputs):
        x = self.encoder(inputs)
        # Apply PyramidDilatedConv to the last output of the encoder
        x = list(x)
        #print("Encoder outputs:", [o.shape for o in x])
        x[-1] = self.res2se_512(x[-1])  # Assuming
        #x[-1] = self.PDC(x[-1])  # Assuming x[-1] is the last output from the encoder
        x[-1] = self.res2se_512(x[-1])  # Apply Res2SEBlock to the last output
        seg = self.seg_decoder(x)
        class_out = self.class_decoder(x)
        return {
            'classification': class_out,
            'segmentation': seg
        }
    
    def build(self, input_shape):
        # Build the model with the specified input shape
        inputs = tf.keras.Input(shape=input_shape[1:])
        self.call(inputs)
        super(MRPUNetMT, self).build(input_shape)

if __name__ == "__main__":
    input_shape = (1, 576, 576, 3)  # Example input shape

    sample_input = tf.random.normal(input_shape)
    model = MRPUNetMT(num_classes=8, num_seg_classes=1)  # Example number of classes
    model.build((None, 576, 576, 3))  # Build the model with the input shape
    output = model(sample_input)

    print("Input shape:", sample_input.shape)
    print("---------------")
    if isinstance(output, tuple):
        for i, o in enumerate(output):
            print(f"Output {i+1} shape: {o.shape}")
    elif isinstance(output, dict):
        for key, value in output.items():
            print(f"{key} shape: {value.shape}")
    else:
        print(f"Output shape: {output.shape}")
    print("---------------")
    

    

    # #Get a test input image 
    # img_path = r"/home/adamdowse/PhD/SkinLesion/Data/HAM10000/train/akiec/ISIC_0024463.jpg"
    # x = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
    # x = tf.image.resize(x, (576, 576))
    # x = tf.expand_dims(x, axis=0)  # Add batch dimension
    # x = x / 255.0  # Normalize the image
    # hr = HairRemoval(kernel_size=10, threshold=0.5)
    # hr.build(input_shape=x.shape)
    # ouptut = hr(x)
    # print("Output shape after hair removal:", ouptut.shape)

    # new_img = remove_hairs_apc_like(x)
    # print("New image shape after hair removal:", new_img.shape)


    # fig = plt.figure(figsize=(10, 5))
    # ax1 = fig.add_subplot(1, 4, 1)
    # ax1.imshow(tf.squeeze(x))
    # ax2 = fig.add_subplot(1, 4, 2)
    # ax2.imshow(tf.squeeze(ouptut))  
    # fig.suptitle("Hair Removal Output")
    # ax3 = fig.add_subplot(1, 4, 3)
    # ax3.imshow(tf.squeeze(new_img))

    # plt.tight_layout()
    # fig.savefig("hair_removal_output.png")
    
