import os
import shutil
import pandas as pd

import argparse

def split_custom_dataset(
        imgs_dir, pose_info_path, out_path,
        val_frac = 0, test_frac = 0, shuffle = False,
        seed = None,
    ):
    """
    Splits the custom dataset into train, val and test splits.
    
    Args:
        imgs_dir        :   Path to the directory containing 
                            the images.
        pose_info_path  :   Path to the pose info csv file.
        out_path        :   Path to the directory which would 
                            contain the output. 
        val_frac        :   A float between 0 and 1 representing 
                            the fraction of the dataset that is 
                            to be used for the validation split.
        test_frac       :   A float between 0 and 1 representing 
                            the fraction of the dataset that is 
                            to be used for the test split.
        shuffle         :   A boolean value which indicates whether 
                            the pose info DataFrame has to be shuffled 
                            before splitting.
        seed            :   The seed that is passed to random_state 
                            argument of df.sample if shuffle is True.
    """
    assert val_frac > 0 and val_frac < 1
    assert test_frac > 0 and test_frac < 1
    train_frac = 1 - (val_frac + test_frac)

    df = pd.read_csv(pose_info_path)
    count = len(df)

    num_train =  int(count * train_frac)
    num_test = int(count * test_frac)
    num_val = count - (num_train + num_test)

    if shuffle:
        df = df.sample(frac = 1, random_state = seed).reset_index(drop = True)

    train_df = df.iloc[:num_train]
    val_df = df.iloc[num_train : (num_train + num_val)]
    test_df = df.iloc[(num_train + num_val):]

    train_df = train_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)
    val_df = val_df.reset_index(drop = True)

    train_imgs_dir = os.path.join(out_path, "train")
    test_imgs_dir = os.path.join(out_path, "test")
    val_imgs_dir = os.path.join(out_path, "val")
    train_pose_info_path = os.path.join(out_path, "train_pose_info.csv")
    test_pose_info_path = os.path.join(out_path, "test_pose_info.csv")
    val_pose_info_path = os.path.join(out_path, "val_pose_info.csv")

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok = True)

    if not os.path.exists(train_imgs_dir):
        os.makedirs(train_imgs_dir, exist_ok = True)

    if not os.path.exists(val_imgs_dir):
        os.makedirs(val_imgs_dir, exist_ok = True)

    if not os.path.exists(test_imgs_dir):
        os.makedirs(test_imgs_dir, exist_ok = True)

    train_df.to_csv(train_pose_info_path, index = False)
    test_df.to_csv(test_pose_info_path, index = False)
    val_df.to_csv(val_pose_info_path, index = False)

    _copy_files(train_df, imgs_dir, train_imgs_dir)
    _copy_files(test_df, imgs_dir, test_imgs_dir)
    _copy_files(val_df, imgs_dir, val_imgs_dir)


def _copy_files(df, src_dir, dst_dir):
    """
    Copies the images which are mentioned in the 
    given DataFrame from the given source directory 
    to the given destination directory.
    """
    for row in df.itertuples():
        filename = row.image_name
        src_path =  os.path.join(src_dir, filename)
        dst_path =  os.path.join(dst_dir, filename)
        shutil.copyfile(src_path, dst_path)


def get_args():
    """
    Gets args from argparse.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", required=True, type=str)
    parser.add_argument("--pose_info_path", required=True, type=str)
    parser.add_argument("--out_path", required=True, type=str)
    parser.add_argument("--val_frac", required=True, type=float)
    parser.add_argument("--test_frac", required=True, type=float)
    parser.add_argument("--shuffle", required=True, action="store_true")
    parser.add_argument("--seed", type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = get_args()

    split_custom_dataset(
        imgs_dir = args.imgs_dir,
        pose_info_path = args.pose_info_path,
        out_path = args.out_path,
        val_frac = args.val_frac,
        test_frac =  args.test_frac,
        shuffle =  args.shuffle,
        seed = args.seed,
    )
