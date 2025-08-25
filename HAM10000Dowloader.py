import numpy as np
import pandas as pd
import os
import scipy as sp
import shutil
import tensorflow as tf
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from sklearn import model_selection
from tqdm import tqdm


# Path: HAM10000Dowloader.py


def download_data():
    data_urls = ['https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T#']
    extract_path = '/tmp/HAM10000'
    for url in data_urls:
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(extract_path)
    print(f"Data downloaded and extracted to {extract_path}")
    #unzip the image files

def calc_length():
    #show the size of each directory
    data_path = "/home/adamdowse/PhD/SkinLesion/Data/HAM10000"
    images_path = os.path.join(data_path, "Images")
    segmaps_path = os.path.join(data_path, "SegmentationMaps")
    images_count = len(os.listdir(images_path))
    segmaps_count = len(os.listdir(segmaps_path))
    print(f"Number of images: {images_count}")
    print(f"Number of segmentation maps: {segmaps_count}")


def extract_data():
    data_path = "/home/adamdowse/PhD/SkinLesion/Data"
    extract_path = '/home/adamdowse/PhD/SkinLesion/Data/HAM10000'
    root_file_dir = os.path.join(data_path, "dataverse_files.zip")
    if not os.path.exists(root_file_dir):
        print(f"File not found: {root_file_dir}")
        return
    with ZipFile(root_file_dir, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Data files extracted.")

    image_paths = [os.path.join(extract_path, f) for f in ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]]
    for image_path in image_paths:
        with ZipFile(image_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(extract_path, "Images"))
    print("Image files extracted.")

    #unzip the segmentation maps
    segmap_path = os.path.join(extract_path, "HAM10000_segmentations_lesion_tschandl.zip")
    with ZipFile(segmap_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(extract_path, "SegmentationMaps"))
    # move the HAM10000_segmentations_lesion_tschandl directory to SegmentationMaps and delete the original directory + __MACOSX
    segmap_dir = os.path.join(extract_path, "SegmentationMaps", "HAM10000_segmentations_lesion_tschandl")
    if os.path.exists(segmap_dir):
        for item in os.listdir(segmap_dir):
            s = os.path.join(segmap_dir, item)
            d = os.path.join(extract_path, "SegmentationMaps", item)
            if os.path.isdir(s):
                shutil.move(s, d)
            else:
                shutil.copy2(s, d)
        shutil.rmtree(segmap_dir)
    # remove the _MACOSX directory if it exists
    macosx_dir = os.path.join(extract_path, "SegmentationMaps", "__MACOSX")
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir)



def seperate_train_test(remove_duplicates=True):
    data_path = "/home/adamdowse/PhD/SkinLesion/Data/HAM10000"

    metadata = pd.read_csv(os.path.join(data_path, "HAM10000_metadata.csv"))
    
    if remove_duplicates:
        #deal with duplicates so there is only one image per lesion_id
        metadata = metadata.drop_duplicates(subset='lesion_id', keep='first')
    # Split the data into train and test sets
    train_MD, test_MD = model_selection.train_test_split(metadata, test_size=0.15, stratify=metadata['dx'])

    # Create directories for train and test sets
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # Create subdirectories for each class
    targetnames = metadata['dx'].unique()
    for target in targetnames:
        os.makedirs(os.path.join(train_dir, target), exist_ok=True)
        os.makedirs(os.path.join(test_dir, target), exist_ok=True)
    # Copy images to train and test directories
    train_image_ids = train_MD['image_id'].tolist()
    train_classes = train_MD['dx'].tolist()
    test_image_ids = test_MD['image_id'].tolist()
    test_classes = test_MD['dx'].tolist()
    print(f"Number of train images: {len(train_image_ids)}")
    print(f"Number of test images: {len(test_image_ids)}")
    # count the number of images in the Images directory
    images_dir = os.path.join(data_path, "Images")
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    total_images = len(os.listdir(images_dir))
    print(f"Total number of images in the Images directory: {total_images}")

    train_not_found = 0
    for image_id, image_class in tqdm(zip(train_image_ids, train_classes)):
        source_path = os.path.join(data_path, "Images", image_id + ".jpg")
        target_path = os.path.join(train_dir, image_class, image_id + ".jpg")
        if not os.path.exists(source_path):
            print(f"Image {image_id} not found in source path: {source_path}")
            train_not_found += 1
            # continue
        shutil.copyfile(source_path, target_path)
    print(f"Number of images not found in train set: {train_not_found}")
    test_not_found = 0
    for image_id, image_class in tqdm(zip(test_image_ids, test_classes)):
        source_path = os.path.join(data_path, "Images", image_id + ".jpg")
        target_path = os.path.join(test_dir, image_class, image_id + ".jpg")
        if not os.path.exists(source_path):
            test_not_found += 1
            continue
        shutil.copyfile(source_path, target_path)
    print(f"Number of images not found in test set: {test_not_found}")

def remove_train_test_dirs():
    data_path = "/home/adamdowse/PhD/SkinLesion/Data/HAM10000"
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("Train and test directories removed.")

def process_data():
    #REMEMBER TO CHANGE THE PATH
    data_path = "Data/HAM10000"
    train_dir = "HAM10000/train"
    test_dir = "HAM10000/test"
    aug_dir = 'HAM10000/aug_dir' #this directory is temporary and will be deleted after the augmented images are generated
    
    data_pd = pd.read_csv("HAM10000/HAM10000_metadata.csv")

    df_count = data_pd.groupby('lesion_id').count()
    df_count = df_count[df_count['dx'] == 1]
    df_count.reset_index(inplace=True)

    def duplicates(x):
        unique = set(df_count['lesion_id'])
        if x in unique:
            return 'no' 
        else:
            return 'duplicates'
    
    data_pd['is_duplicate'] = data_pd['lesion_id'].apply(duplicates)
    data_pd.head()

    df_count = data_pd[data_pd['is_duplicate'] == 'no']
    train, test_df = model_selection.train_test_split(df_count, test_size=0.15, stratify=df_count['dx'])

    def identify_trainOrtest(x):
        test_data = set(test_df['image_id'])
        if str(x) in test_data:
            return 'test'
        else:
            return 'train'

    #creating train_df
    data_pd['train_test_split'] = data_pd['image_id'].apply(identify_trainOrtest)
    train_df = data_pd[data_pd['train_test_split'] == 'train']

    # Image id of train and test images
    train_list = list(train_df['image_id'])
    test_list = list(test_df['image_id'])

    print('Train DF sise: ', len(train_list))
    print(train_df.head())
    print('Test DF size: ', len(test_list))
    print(test_df.head())

    # Set the image_id as the index in data_pd
    data_pd.set_index('image_id', inplace=True)

    #Build file structure
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for i in targetnames:
        directory1=train_dir+'/'+i
        directory2=test_dir+'/'+i
        os.mkdir(directory1)
        os.mkdir(directory2)

    print('Copying images into train and test directories')
    for image in train_list:
        file_name = image+'.jpg'
        label = data_pd.loc[image, 'dx']

        # path of source image 
        source = os.path.join('/com.docker.devenvironments.code/HAM10000/HAM', file_name)

        # copying the image from the source to target file
        target = os.path.join(train_dir, label, file_name)

        shutil.copyfile(source, target)

    for image in test_list:

        file_name = image+'.jpg'
        label = data_pd.loc[image, 'dx']

        # path of source image 
        source = os.path.join('/com.docker.devenvironments.code/HAM10000/HAM', file_name)

        # copying the image from the source to target file
        target = os.path.join(test_dir, label, file_name)

        shutil.copyfile(source, target)

    # this is the dir for the reduced ds 
    reduced_dir = 'HAM10000/reduced/train'
    os.mkdir('HAM10000/reduced')
    os.mkdir(reduced_dir)
    for i in targetnames:
        directory1=reduced_dir+'/'+i
        os.mkdir(directory1)

    # Augmenting images and storing them in temporary directories 
    for img_class in targetnames:

        #creating temporary directories
        # creating a base directory
        print('Augmenting class: ', img_class)
        os.mkdir(aug_dir)
        # creating a subdirectory inside the base directory for images of the same class
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.mkdir(img_dir)

        img_list = os.listdir('HAM10000/train/' + img_class)

        # Copy images from the class train dir to the img_dir 
        for file_name in img_list:
            # path of source image in training directory
            source = os.path.join('HAM10000/train/' + img_class, file_name)

            # creating a target directory to send images 
            target = os.path.join(img_dir, file_name)

            # copying the image from the source to target file
            shutil.copyfile(source, target)

        # Temporary augumented dataset directory.
        source_path = aug_dir

        # Augmented images will be saved to training directory
        save_path = 'HAM10000/reduced/train/' + img_class

        # Creating Image Data Generator to augment images
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        batch_size = 50

        aug_datagen = datagen.flow_from_directory(source_path,save_to_dir=save_path,save_format='jpg',save_prefix='aug',target_size=(299, 299),batch_size=batch_size,shuffle=True)

        # Generate the augmented images
        aug_images = 300

        #force save all generated images
        if False:
            num_files = len(os.listdir(img_dir))
            num_batches = int(np.ceil((aug_images - num_files) / batch_size))
        else:
            num_batches = int(np.ceil(aug_images / batch_size))

        # creating 8000 augmented images per class
        for i in range(0, num_batches):
            images, labels = next(aug_datagen)

        # delete temporary directory 
        shutil.rmtree('HAM10000/aug_dir')
    
    
if __name__ == "__main__":
    #download_data()
    #extract_data()
    #calc_length()
    #process_data()
    #remove_train_test_dirs()
    seperate_train_test(remove_duplicates=False)







