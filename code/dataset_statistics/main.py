# fixed parameters
# - train_dataset_dir
# - annotation_dir

# tunable parameters:
# - window sliding mode: fixed anchor or variable anchor.
#   - fixed anchor: window sliding as convolution fashion.
#   - variable anchor: window chosen to cover the most densely populated area with the help of oracle.
#     - this includes a lot of optimizations and dynamic programming. Place this as a question mark.
# - sliding window length
# - sliding window overlap percentage (related to the maximal size of desired objects)
# - discard policy percentage (eg. 0.01 for: if less than 1% of the area is covered inside the bounding box, 
#   then this subimage is considered as of no interest)

# outputs: 
# - percentage of discarded sub images (mean and variance over the training set).
# - percentage of discarded sub images - outliers (3 sigma, with image name).
# - bounding box coverage (mean and variance over the training set).
# - bounding box coverage - outliers (3 sigma, with image name).

