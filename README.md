# Object-aware-Video-Style-Transfer-Network-with-Long-Short-Temporal-and-Depth-Consistent-Constraints

## Abstract
Video style transfer, as a natural extension of image style transfer, has recently gained much interests. However, existing image based methods cannot be readily extended to videos because of temporal flickering and stylization inconsistency.Therefore, the main effort of this work is to propose an efficient salient object-aware and depth-consistent video style transfer algorithm. Specifically, we carefully extenddensenet as feed-forward backbone network for better style transfer quality. Then, through utilizing salient object segmentation and depth estimation results, we deliberately propose depth-consistent loss and object masked long-short temporal losses for learning control at training stage. The proposed losses can preserve stereoscopic sense without salient semantic distortion and consecutive stylized frame flickering. We have compared our proposed network with several state-of-the-art methods. The experimental results demonstrate that our method is more superior on achieving realtime processing efficiency, nice rendering quality, and coherent stylization at the same time.

## Prerequisites
We recommend Anaconda as the environment


Python : 3.6 


Pytorch : 1.4.0


Ubuntu : 18.04

## Prepare Net
DepthNet:[github link](https://github.com/A-Jacobson/Depth_in_The_Wild)


BASNet:[github link](https://github.com/NathanUA/BASNet)

## Training
Please set the datasert path and other net path in Class Config from main.py
```
python main.py
```

## Test
Please commented out train() and uncomment out stylize() in main.py

