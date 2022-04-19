# Detailed Planning till Early June

## Additional

I will work @ ESAT and document the progress everyday. 

### April 18 to April 24

- Test the effect of changing of resolution. On the pre-trained model.
- Statistics of the original training set data, including:
  - % of object coverage using different cropping size, overlapping ratio
  - test both for sliding window approach and flexibly-defined bounding box anchors (maybe the optimal flexibly-defined bounding box positions based on cropping size + overlapping ratio)
- write the thesis text for 1) scarcity characteristics of the dataset and 2) accuracy impact over change of input resolution (of the pre-trained model)
- discuss the results with Thomas about feasible acceleration strategies, then proceed.
- continue working on the alignment bug at the same time.
  - for input of cropped sub images, desired output: exact bounding box predictions as the result of the pre-trained model with original input image.
  - the necessity of implementing 'flexibly-defined bounding box' is to be discussed after the statistics is obtained. 
- writeups for:
  - Is the aassumption "images are dense in a smaller scale and sparse in a large scale" correct?

### April 25 to May 1

- Evaluation of accuracy and time lapse over different setups:
  - different downsampling ratios (with a fixed subimage size as the original network) while keeping the checkpoint file same as original pre-trained model.
  - different downsampling ratios (with a fixed subimage size as the original network) while retraining the model with model whose depth is 51 and 101.
  - different subimage sizes (after this option, a group of one-image-in-all rather than subimage-related statistics is geenrated)
- writeup of this part, regarding to these questions:
  - what is the relation among object size statistics, window size, window overlapping ratio, and a network optimized with these following parameters?
- discussion, and the next steps.

### May 2 to May 8

### May 9 to May 15

### May 16 to May 22

### May 23 to May 29

### May 30 to June 5