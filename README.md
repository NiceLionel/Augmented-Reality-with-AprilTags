## Augmented Reality Application Using AprilTags

Welcome to my Augmented Reality (AR) project leveraging the robustness of AprilTags for camera pose estimation!

### Introduction

Augmented Reality (AR) has been a game-changing technology in various fields, adding a layer of digital information to our physical world. By overlaying virtual 3D objects onto real-world scenes, AR applications create immersive experiences for users. In this project, I aim to realize a simple yet effective AR application that overlays virtual objects onto a video, making it seem as if these objects coexist with the real world.

I utilize a video where each frame consists of an [AprilTag](https://april.eecs.umich.edu/software/apriltag) - a widely recognized tool in robotics for estimating the camera's pose with high precision. Thanks to its design and algorithm, AprilTags allow me to pinpoint the camera's position and orientation with respect to the tag. 

In this project, I've also been provided with the four corner coordinates and the size of these tags. With this information, my primary goal is to recover the camera poses using two distinct approaches:

1. **Perspective-N-Point (PnP) with Coplanar Assumption**: By utilizing the 2D-3D correspondences, this method estimates the pose of the camera.
2. **Perspective-Three-Point (P3P) & Procrustes Problem**: Another powerful technique that leverages minimalistic 2D-3D point correspondences to estimate the camera pose, followed by the Procrustes analysis for alignment.



### Usage
```
cd code
python main.py
```

## Customization

You need to assign different values to  `click_point` in `main.py` to render the drill at different places. 



