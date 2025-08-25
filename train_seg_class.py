#train the MRP-Unet model for segmentation tasks on skin lesion images.



import tensorflow as tf
import Models as Models
import os
import wandb
import pandas as pd
from tqdm import tqdm

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint



def create_segmentation_dataset_balanced(data_dir, image_size=(576, 576), batch_size=2, num_classes=7, train=True):
    """
    Creates a tf.data.Dataset for image segmentation and classification, balanced by the classification label.

    This function uses resampling to handle class imbalance. It creates a separate
    dataset for each class and then samples from them with equal probability.

    Args:
        data_dir (str): The root directory of the dataset.
        image_size (tuple): The target size (height, width) for the images and masks.
        batch_size (int): The number of samples per batch.
        train (bool): Whether to use the 'train' or 'test' subdirectory.

    Returns:
        A balanced tf.data.Dataset object.
    """
    if train:
        images_dir = os.path.join(data_dir, 'train')
    else:
        images_dir = os.path.join(data_dir, 'test')

    masks_dir = os.path.join(data_dir, 'SegmentationMaps')

    images_dict = {"id": [], "image_path": [], "mask_path": [], "dx": []}
    for c in tf.io.gfile.listdir(images_dir):
        imgs_in_class = tf.io.gfile.glob(f"{images_dir}/{c}/*.jpg")
        for img_path in imgs_in_class:
            img_name = os.path.basename(img_path).replace('.jpg', '')
            mask_path = os.path.join(masks_dir, f"{img_name}_segmentation.png")

            # Ensure both the image and its corresponding mask exist
            if tf.io.gfile.exists(mask_path):
                images_dict["id"].append(img_name)
                images_dict["image_path"].append(img_path)
                images_dict["mask_path"].append(mask_path)
                images_dict["dx"].append(c)
    
    # Convert the dictionary to a DataFrame for easier manipulation
    meta_data = pd.DataFrame(images_dict)

    # resample the dataset to balance classes
    #count the number of samples in each class
    class_counts = meta_data['dx'].value_counts()
    if train:
        meta_data = meta_data.groupby('dx').apply(lambda x: x.sample(meta_data['dx'].value_counts().max(), replace=True)).reset_index(drop=True)
    
    print(f"Before resampling, class counts:\n{class_counts} \n After resampling, class counts:\n{meta_data['dx'].value_counts()}")

    # Convert the classification labels to indices
    label_to_index = {label: index for index, label in enumerate(meta_data['dx'].unique())}
    meta_data['label_index'] = meta_data['dx'].map(label_to_index)


    #shuffle the rows
    meta_data = meta_data.sample(frac=1).reset_index(drop=True)

    #make a tf dataset from the DataFrame
    dataset = tf.data.Dataset.from_tensor_slices((meta_data['image_path'].values,
                                                  meta_data['mask_path'].values,
                                                  meta_data['label_index'].values))
    def _load_and_preprocess(img_path, mask_path, label_index):
        # Load image and mask files
        img = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
        mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)

        # Resize and normalize
        img = tf.image.resize(img, image_size)
        mask = tf.image.resize(mask, image_size)
        img = img / 255.0
        img = Models.remove_hairs_apc_like(img)  # Apply hair removal preprocessing
        img.set_shape((image_size[0], image_size[1], 3))  # Ensure correct shape

        # For segmentation, a binary mask is often sufficient (lesion vs. background)
        # Ensure the mask is in the format [height, width, 1]
        mask = tf.cast(mask > 0, tf.float32)

        # One-hot encode the classification label
        classification_output = tf.one_hot(label_index, depth=num_classes, dtype=tf.float32)

        return img, {'classification': classification_output, 'segmentation': mask}
    
    # Map the loading function to the dataset
    dataset = dataset.map(lambda img, mask, label: _load_and_preprocess(img, mask, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    # Shuffle and batch the dataset
    dataset = dataset.batch(batch_size)
    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size = 1)
    return dataset

IoU = tf.keras.metrics.BinaryIoU(
    target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None
)
    
if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    config = {
        "learning_rate": 0.001,
        "batch_size": 2,
        "image_size": (576, 576),
        "num_classes": 1,
        "epochs": 10,
        "model_name": "MRPUNetClass"
    }
    wandb.init(project="SkinMask", name="MRPUNetClass", config= config)
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        print(f"GPUs available: {gpus}")
    else:
        print("No GPUs available.")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # Create a data loader instance
    # data_loader = SegmentationDataLoader(root_dir='/workspace/Data/HAM10000', 
    #                                      batch_size=2, 
    #                                      image_size=(576, 576))
    train_dataset = create_segmentation_dataset_balanced(data_dir='/home/adamdowse/PhD/SkinLesion/Data/HAM10000',
                                                image_size=(576, 576),
                                                batch_size=1,
                                                num_classes=7,
                                                train=True)
    # test_dataset = create_segmentation_dataset(data_dir='/workspace/Data/HAM10000',
    #                                             image_size=(576, 576),
    #                                             batch_size=2,
    #                                             train=False)


    IoU = tf.keras.metrics.BinaryIoU(
        target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None
    )

    # Create the model
    model = Models.MRPUNetMT(num_seg_classes=1,num_classes=7)  # Assuming binary segmentation
    model.build(input_shape=(None, 576, 576, 3))
    losses = {
        'classification': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        'segmentation': tf.keras.losses.Dice()
    }
    metrics = {
        'classification': 'accuracy',
        'segmentation': ['accuracy',IoU] # For segmentation, you might want a more advanced metric like MeanIoU
    }
    weights = {
        'classification': 0.5,
        'segmentation': 0.5
    }
    EPOCHS = 10
    #lr should start at 0.001 and end at 0.000001 over the course of training
    # Create a learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,  # Adjust based on your dataset size and batch size
        decay_rate=0.96,
        staircase=False
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer='adam', loss=losses, metrics=metrics, loss_weights=weights)
    model.summary()

    # #test
    # for batch in train_dataset.take(1):
    #     x, y = batch
    #     out = model(x)
    #     print("Batch ran successfully")
    # Train the model
    model.fit(train_dataset,epochs=10, verbose=1,callbacks=[WandbMetricsLogger(log_freq=10), WandbModelCheckpoint(save_freq="epoch",filepath=r"/home/adamdowse/PhD/SkinLesion/Models/MRPUNetMT.keras")]) #, validation_data=test_dataset


    wandb.finish()
    
    