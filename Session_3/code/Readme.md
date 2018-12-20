# Acknowledgements #

## I. Representative images and 1-2 training/testing images for demonstration have been taken from: ##
1. "Stereo Vision: Algorithms and Applications" by Stefano Mattocia
2. Fischer et al., FlowNet: Learning Optical Flow with Convolutional Networks. ICCV 2015
3. Mayer et al., A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation. CVPR (2015)
4. Luo et al., Efficient Deep Learning for Stereo Matching. CVPR 2016
5. Scene Flow: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
6. KITTI 2015: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
7. Middlebury: http://vision.middlebury.edu/stereo
8. ETH3D: https://www.eth3d.net/low_res_two_view


## II. Utility functions for converting pfm files to png (util.py, util_stereo.py and png.py) have been taken from: ##
1. http://www.cvlibs.net:3000/ageiger/rob_devkit


## III. Utility to read and write pfm files is taken from original authors of Scene Flow: ##
1. https://lmb.informatik.uni-freiburg.de/resources/

Direct link: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlow/assets/code/python_pfm.py

The file has been slightly modified to suite our execution environment





# Finally, follow these steps to run the code in jupyter notebook: #
1. Make sure you are inside "Session_3/code" directory and then open "session_3_depth_hands_on.ipynb" notebook 

2. Execute the initial 4-5 lines in the notebook: they are for cloning the git repository (if required) and merging and moving weight files
   - NOTE: Github doesn't allow file of size more than 100 MB, so I had to split each weight file into two parts

3. main.py, model.py and dataloader.py constitute the final merged code, refer them to know how to write a full working code after following the jupyter notebook hands-on. Run following command to learn about required arguments in main.py file:

```
python main.py -h
```

or 

```
python main.py --help
```