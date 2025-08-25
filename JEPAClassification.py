import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 1. HYPERPARAMETERS ---
# These would need to be tuned for your specific dataset
IMAGE_SIZE = 128
NUM_CHANNELS = 3
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
LATENT_DIM = 256  # Dimension of the latent space from the encoder
NUM_HEADS = 4     # For the transformer blocks
NUM_CLASSES = 7   # e.g., 7 types of skin cancer
LEARNING_RATE = 0.001

# Loss weights - crucial for balancing the three tasks
LOSS_WEIGHT_CLASSIFICATION = 1.0  # alpha
LOSS_WEIGHT_RECONSTRUCTION = 0.25 # beta
LOSS_WEIGHT_SEGMENTATION = 0.75   # gamma


# --- 2. ENCODER ARCHITECTURE (ViT-inspired) ---
# In a real scenario, this encoder would be pre-trained using a JEPA objective.

def get_encoder():
    """Builds a patch-based Vision Transformer style encoder."""
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    
    # 1. Patching
    # Divide the image into patches
    patches = tf.image.extract_patches(
        images=inputs,
        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patch_dims = patches.shape[-1]
    patches = layers.Reshape((NUM_PATCHES, patch_dims))(patches)

    # 2. Patch & Position Embedding
    # Project patches to the latent dimension and add position information
    patch_embedding = layers.Dense(units=LATENT_DIM)(patches)
    positions = tf.range(start=0, limit=NUM_PATCHES, delta=1)
    position_embedding = layers.Embedding(input_dim=NUM_PATCHES, output_dim=LATENT_DIM)(positions)
    encoded_patches = patch_embedding + position_embedding

    # 3. Transformer Blocks
    # This is where the model learns relationships between patches
    for _ in range(4): # Number of transformer blocks
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=LATENT_DIM // NUM_HEADS, dropout=0.1
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = layers.Dense(LATENT_DIM * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(LATENT_DIM)(x3)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    
    # 4. Final Representation
    # We take the global average of all patch representations
    representation = layers.GlobalAveragePooling1D()(encoded_patches)
    
    # Create the Keras Model for the encoder
    encoder = keras.Model(inputs=inputs, outputs=[representation, encoded_patches])
    return encoder


# --- 3. DECODER "HEADS" ---

def get_decoder_heads(latent_dim, num_patches, patch_size):
    """Builds the three output heads for the model."""
    
    # Input Tensors
    latent_representation = layers.Input(shape=(latent_dim,), name="latent_rep_input")
    patch_representations = layers.Input(shape=(num_patches, latent_dim), name="patches_rep_input")
    
    # --- Classification Head ---
    # Takes the global representation and predicts the class
    clf_head = layers.Dense(latent_dim // 2, activation='relu')(latent_representation)
    clf_head = layers.Dropout(0.5)(clf_head)
    clf_output = layers.Dense(NUM_CLASSES, activation='softmax', name='classification')(clf_head)

    # --- Reconstruction & Segmentation Heads (using Transposed Convolutions) ---
    # Reshape the patch representations back into a 2D feature map
    num_feat_maps = IMAGE_SIZE // patch_size
    reshaped_patches = layers.Reshape((num_feat_maps, num_feat_maps, latent_dim))(patch_representations)
    
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(reshaped_patches)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    
    # Final upsampling to the original image size
    final_upsample = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)

    # --- Reconstruction Head Output ---
    # Reconstructs the original image
    recon_output = layers.Conv2D(
        NUM_CHANNELS, 3, padding='same', activation='sigmoid', name='reconstruction'
    )(final_upsample)
    
    # --- Segmentation Head Output ---
    # Generates a 1-channel segmentation mask
    seg_output = layers.Conv2D(
        1, 3, padding='same', activation='sigmoid', name='segmentation'
    )(final_upsample)

    # Create the Keras Model for the decoder heads
    decoder = keras.Model(
        inputs=[latent_representation, patch_representations],
        outputs=[clf_output, recon_output, seg_output]
    )
    return decoder


# --- 4. THE FULL MULTI-TASK MODEL ---

class JepaMultiTaskModel(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def call(self, inputs):
        # The forward pass: image -> encoder -> decoder heads
        latent_representation, patch_representations = self.encoder(inputs)
        clf, recon, seg = self.decoder([latent_representation, patch_representations])
        return {"classification": clf, "reconstruction": recon, "segmentation": seg}

    def compile(self, optimizer, loss_fns, loss_weights):
        super().compile(optimizer=optimizer)
        self.loss_fns = loss_fns
        self.loss_weights = loss_weights
        
    def train_step(self, data):
        # Unpack the data. It must be a tuple of (inputs, targets).
        # Targets are a dictionary matching the output names.
        image, targets = data
        y_clf = targets['classification']
        y_recon = targets['reconstruction']
        y_seg = targets['segmentation']
        
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(image, training=True)
            
            # Compute individual losses
            loss_clf = self.loss_fns['classification'](y_clf, outputs['classification'])
            loss_recon = self.loss_fns['reconstruction'](y_recon, outputs['reconstruction'])
            loss_seg = self.loss_fns['segmentation'](y_seg, outputs['segmentation'])
            
            # Compute the total weighted loss
            total_loss = (loss_clf * self.loss_weights['classification'] +
                          loss_recon * self.loss_weights['reconstruction'] +
                          loss_seg * self.loss_weights['segmentation'])

        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Return a dict of metrics to track
        return {
            "total_loss": total_loss,
            "classification_loss": loss_clf,
            "reconstruction_loss": loss_recon,
            "segmentation_loss": loss_seg,
        }

    def test_step(self, data):
        # Same logic as train_step, but without gradient updates
        image, targets = data
        y_clf = targets['classification']
        y_recon = targets['reconstruction']
        y_seg = targets['segmentation']
        
        outputs = self(image, training=False)
        
        loss_clf = self.loss_fns['classification'](y_clf, outputs['classification'])
        loss_recon = self.loss_fns['reconstruction'](y_recon, outputs['reconstruction'])
        loss_seg = self.loss_fns['segmentation'](y_seg, outputs['segmentation'])
        
        total_loss = (loss_clf * self.loss_weights['classification'] +
                      loss_recon * self.loss_weights['reconstruction'] +
                      loss_seg * self.loss_weights['segmentation'])
        
        return {
            "total_loss": total_loss,
            "classification_loss": loss_clf,
            "reconstruction_loss": loss_recon,
            "segmentation_loss": loss_seg,
        }

# --- 5. USAGE EXAMPLE ---

if __name__ == '__main__':
    # 1. Instantiate the model components
    encoder = get_encoder()
    decoder = get_decoder_heads(LATENT_DIM, NUM_PATCHES, PATCH_SIZE)
    model = JepaMultiTaskModel(encoder, decoder)

    # 2. Define losses and weights
    # Note: Using Dice loss for segmentation is often better than MSE/MAE
    # For simplicity, we use common losses here.
    loss_functions = {
        "classification": keras.losses.CategoricalCrossentropy(),
        "reconstruction": keras.losses.MeanSquaredError(),
        "segmentation": keras.losses.Dice(), # Or a Dice Loss function
    }
    loss_weights = {
        "classification": LOSS_WEIGHT_CLASSIFICATION,
        "reconstruction": LOSS_WEIGHT_RECONSTRUCTION,
        "segmentation": LOSS_WEIGHT_SEGMENTATION,
    }

    # 3. Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss_fns=loss_functions,
        loss_weights=loss_weights
    )

    # 4. Create dummy data to simulate training
    # **IMPORTANT**: Your real data loader must yield data in this format!
    batch_size = 8
    dummy_images = tf.random.normal((batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    dummy_labels = tf.one_hot(tf.random.uniform((batch_size,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32), depth=NUM_CLASSES)
    dummy_masks = tf.random.uniform((batch_size, IMAGE_SIZE, IMAGE_SIZE, 1))
    
    # The target data must be a dictionary with keys matching the output layer names
    dummy_targets = {
        "classification": dummy_labels,
        "reconstruction": dummy_images, # The model tries to reconstruct the input
        "segmentation": dummy_masks
    }

    print("--- Training the model on a dummy batch ---")
    history = model.fit(dummy_images, dummy_targets, epochs=2, batch_size=batch_size)
    print("\nTraining History:", history.history)

    # 5. Inference Example
    print("\n--- Running inference on a single image ---")
    test_image = tf.random.normal((1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    predictions = model.predict(test_image)
    
    predicted_class = tf.argmax(predictions['classification'], axis=-1).numpy()
    reconstructed_image = predictions['reconstruction'][0]
    segmentation_mask = predictions['segmentation'][0]

    print(f"Predicted Class Index: {predicted_class[0]}")
    print(f"Reconstructed Image Shape: {reconstructed_image.shape}")
    print(f"Segmentation Mask Shape: {segmentation_mask.shape}")
    
    # You can get a summary of the internal models
    model.encoder.summary()
    model.decoder.summary()