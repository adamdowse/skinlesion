#test the feature importance of our generated features and save the resutls 

import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_image(row, data_path):
    """
    Process a single image and calculate features.
    """
    image_path = f"{data_path}/Images/{row['image_id']}.jpg"
    mask_path = f"{data_path}/SegmentationMaps/{row['image_id'].split('.')[0]}_segmentation.png"
    image = plt.imread(image_path)
    mask = plt.imread(mask_path)
    if len(mask.shape) != 2:
        # If the mask has an alpha channel or is RGB, convert it to grayscale
        if mask.shape[-1] == 4:
            mask = mask[:, :, 0]
        elif mask.shape[-1] == 3:
            mask = np.mean(mask, axis=-1)
    
    mask = mask > 0.5
    #do checks 
    if image is None or mask is None:
        print(f"Image or mask not found for image_id {row['image_id']}. Skipping.")
        return None
    if image.shape[:2] != mask.shape[:2]:
        print(f"Image and mask dimensions do not match for image_id {row['image_id']}. Skipping.")
        return None
    
    f = {}
    f["image_id"] = row["image_id"]
    f["asymmetry_iou"] = utils.asymmetryIoU(mask, draw=False)
    f["asymmetry_colour"] = utils.asymmetryColour(mask, image, draw=False)
    f["compact_index"] = utils.compactIndex(mask, draw=False)
    f["fractal_dimension"] = utils.fracalDimension(mask, draw=False)
    f["border_gradient"] = utils.borderGradient(mask, image, draw=False)
    popt = utils.fitted_gaussian(mask, image, draw=False)
    if popt is None:
        print(f"Fitted Gaussian failed for image {row['image_id']}. Using default values.")
        popt = [0, 0, 0, 1, 1, 0, 0]
    f["gaussian_amplitude"] = popt[0]
    f["gaussian_x0"] = popt[1]
    f["gaussian_y0"] = popt[2]
    f["gaussian_sigma_x"] = popt[3]
    f["gaussian_sigma_y"] = popt[4]
    f["gaussian_theta"] = popt[5]
    f["gaussian_offset"] = popt[6]
    return f

def calc_features():
    
    # load the metadata
    data_path = "/com.docker.devenvironments.code/SkinLesion/Data/HAM10000"
    metadata = pd.read_csv(f"{data_path}/HAM10000_metadata.csv")

    # Use multiprocessing to process images
    n_images = len(metadata)
    print(f"Calculating features for {n_images} images...")
    features = []
    for index, row in tqdm(metadata.iterrows(),total=n_images, desc="Processing images"):
        features.append(process_image(row, data_path))



    # Save features to a CSV file
    features_df = pd.DataFrame(features)
    features_df.set_index("image_id", inplace=True)
    features_df.to_csv(f"{data_path}/features.csv")
    print("Features calculated and saved to features.csv")

def feature_importance():
    data_path = "/com.docker.devenvironments.code/SkinLesion/Data/HAM10000"
    metadata = pd.read_csv(f"{data_path}/HAM10000_metadata.csv")
    features_df = pd.read_csv(f"{data_path}/features.csv", index_col="image_id")
    # join the features with the metadata
    metadata.set_index("image_id", inplace=True)
    combined_df = metadata.join(features_df, how="inner")
    print(combined_df.columns)

    # calculate the correlation between the features and the diagnosis
    combined_df["dx"] = combined_df["dx"].astype("category").cat.codes
    # split the border_gradient into two features
    if "border_gradient" in combined_df.columns:
        combined_df[["border_gradient", "border_angle"]] = combined_df["border_gradient"].apply(lambda x: pd.Series([x.split(",")[0].split(")")[0].split("(")[-1],x.split(",")[-1].split(")")[0].split("(")[-1]])).astype(float)
    combined_df = combined_df.drop(["dx_type", "localization", "dataset","image_id","lesion_id","sex"], axis=1, errors='ignore')
    correlation = combined_df.corr()["dx"].drop(["dx"], axis=0)
    correlation = correlation.sort_values(ascending=False)
    print("Feature importance based on correlation with diagnosis:")
    print(correlation)

    # plot the correlation
    plt.figure(figsize=(10, 6))
    correlation.plot(kind='bar')
    plt.title("Feature Importance based on Correlation with Diagnosis")
    plt.xlabel("Features")
    plt.ylabel("Correlation with Diagnosis")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/com.docker.devenvironments.code/SkinLesion/Graphs/feature_importance_correlation.png")

    #xgboost feature importance
    import xgboost as xgb

    X = combined_df.drop("dx", axis=1)
    y = combined_df["dx"]
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'multi:softmax',
        'num_class': len(y.unique()),
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'eta': 0.1,
        'seed': 42
    }
    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    importance = bst.get_score(importance_type='weight')
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("Feature importance based on XGBoost:")
    print(importance_df)
    # plot the xgboost feature importance
    plt.figure(figsize=(10, 6))
    importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False)
    plt.title("Feature Importance based on XGBoost")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/com.docker.devenvironments.code/SkinLesion/Graphs/feature_importance_xgboost.png")

if __name__ == "__main__":
    #calc_features()
    feature_importance()
    print("Feature importance analysis completed.")


