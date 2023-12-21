# Simultaneous-Localization-and-mapping-for-humanoid-robot-THOR


Implemented mapping and localization in an indoor environment using infor-
mation from an IMU and a LiDAR sensor. Data collected
from a humanoid named THOR that was built at Penn and UCLA


Hardware setup of Thor The humanoid has a Hokuyo LiDAR sensor (https://hokuyo-
usa.com/products/lidar-obstacle-detection on its head (the final version of the robot
had it in its chest but this is a different version); details of this are in the code (which
will be explained shortly). This LiDAR is a planar LiDAR sensor and returns 1080
readings at each instant, each reading being the distance of some physical object
along a ray that shoots off at an angle between (-135, 135) degrees with discretiza-
tion of 0.25 degrees in an horizontal plane (shown as white rays in the picture)

We will use the position and orientation of the head of the robot to calculate 
orientation of the LiDAR in the body frame.
