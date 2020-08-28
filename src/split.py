import splitfolders
splitfolders.ratio("../data/train", output="../data/train_split", seed=1337, ratio=(0.8, 0.15, 0.05), group_prefix=None) # default values