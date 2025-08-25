#train the MRP-Unet model for segmentation tasks on skin lesion images.



import tensorflow as tf
import Models as Models
import os
import wandb

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint



class SegmentationDataLoader(tf.keras.utils.Sequence):
    def __init__(self, root_dir, train=True, batch_size=32, image_size=(256, 256)):
        super().__init__()
        self.root_dir = root_dir
        if train:
            self.images_dir = root_dir+ '/train' #/com.docker.devenvironments.code/SkinLesion/Data/HAM10000/train
        else:
            self.images_dir = root_dir + '/test'
        self.masks_dir = root_dir + '/SegmentationMaps' #/com.docker.devenvironments.code/SkinLesion/Data/HAM10000/SegmentationMaps
        self.batch_size = batch_size
        self.image_size = image_size
        self.classes = tf.io.gfile.listdir(self.images_dir)
        self.image_filenames = []
        for c in self.classes:
            # Collect all image filenames for each class
            self.image_filenames.extend(tf.io.gfile.glob(f"{self.images_dir}/{c}/*.jpg"))
        print(f"Found {len(self.image_filenames)} images in {self.images_dir}")
        #ensure a mask exists for each image
        c = 0
        for img in self.image_filenames:
            img_name = img.split('/')[-1].replace('.jpg', '_segmentation.png')
            mask_path = f"{self.masks_dir}/{img_name}"
            if tf.io.gfile.exists(mask_path):
                c += 1
            else:
                print(f"Mask not found for image: {img_name}")

        assert len(self.image_filenames) == c, "Number of images and masks must match"

        # shuffle the image filenames
        self.image_filenames = tf.random.shuffle(self.image_filenames)


    def __len__(self):
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = tf.strings.regex_replace(batch_images, r'/HAM10000/.*/', '/HAM10000/SegmentationMaps/')
        batch_masks = tf.strings.regex_replace(batch_masks, r'\.jpg$', '_segmentation.png')

        images = [tf.image.decode_jpeg(tf.io.read_file(img), channels=3) for img in batch_images]
        masks = [tf.image.decode_png(tf.io.read_file(mask), channels=1) for mask in batch_masks]

        images = [tf.image.resize(img, self.image_size) for img in images]
        masks = [tf.image.resize(mask, self.image_size) for mask in masks]

        # Normalize images to [0, 1] range
        images = [img / 255.0 for img in images]
        # Convert masks to binary (0 or 1)
        masks = [tf.cast(mask > 0, tf.float32) for mask in masks]

        images = tf.stack(images)
        masks = tf.stack(masks)

        return images, masks
    
    def on_epoch_end(self):
        # Shuffle the data at the end of each epoch
        self.image_filenames = tf.random.shuffle(self.image_filenames)

def create_segmentation_dataset(data_dir, image_size=(576, 576), batch_size=2, train=True):
    if train:
        images_dir = os.path.join(data_dir, 'train')
    else:
        images_dir = os.path.join(data_dir, 'test')
    masks_dir = os.path.join(data_dir, 'SegmentationMaps')
    classes = tf.io.gfile.listdir(images_dir)
    image_paths = []
    mask_paths = []
    for c in classes:
        imgs = tf.io.gfile.glob(f"{images_dir}/{c}/*.jpg")
        for img in imgs:
            img_name = img.split('/')[-1].replace('.jpg', '')

            mask_path = os.path.join(masks_dir, f"{img_name}_segmentation.png")
            if tf.io.gfile.exists(mask_path):
                image_paths.append(img)
                mask_paths.append(mask_path)

    def _load_image_mask(img_path, mask_path):
        img = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
        mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)
        img = tf.image.resize(img, image_size)
        mask = tf.image.resize(mask, image_size)
        img = img / 255.0
        mask = tf.cast(mask > 0, tf.float32)
        return img, mask

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(_load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
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
        "model_name": "MRPUNet"
    }
    wandb.init(project="SkinMask", name="MRPUNet", config= config)
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        print(f"GPUs available: {gpus}")
    else:
        print("No GPUs available.")
    # Create a data loader instance
    # data_loader = SegmentationDataLoader(root_dir='/workspace/Data/HAM10000', 
    #                                      batch_size=2, 
    #                                      image_size=(576, 576))
    train_dataset = create_segmentation_dataset(data_dir='/workspace/Data/HAM10000',
                                                image_size=(576, 576),
                                                batch_size=2,
                                                train=True)
    # test_dataset = create_segmentation_dataset(data_dir='/workspace/Data/HAM10000',
    #                                             image_size=(576, 576),
    #                                             batch_size=2,
    #                                             train=False)


    IoU = tf.keras.metrics.BinaryIoU(
        target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None
    )

    # Create the model
    model = Models.MRPUNet(num_classes=1)  # Assuming binary segmentation
    model.build(input_shape=(None, 576, 576, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', IoU])
    model.summary()

    # Train the model
    model.fit(train_dataset,epochs=10, verbose=1,callbacks=[WandbMetricsLogger(log_freq=10), WandbModelCheckpoint(save_freq="epoch",filepath=r"/workspace/Models/MRPUNet.keras")]) #, validation_data=test_dataset


    wandb.finish()
    
    