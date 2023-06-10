# SeeThruFinger
See and Grasp Anything with a Soft Touch

We present SeeThruFinger, a soft robotic finger with an in-finger vision for multi-modal perception, including visual perception and tactile sensing, for geometrically adaptive and real-time reactive grasping. 

This study leverages the Soft Polyhedral Network design as a robotic finger, capable of omni-directional adaptation with an unobstructed view of the finger's spatial deformation from the inside. By embedding a miniature camera underneath, we achieve the visual perception of the external environment by inpainting the finger mask using E2FGV, which can be used for object detection in the downstream tasks for grasping. After contacting the objects, we use real-time object segmentation algorithms, such as XMem, to track the soft finger's spatial deformations. We also learned a Supervised Variational Autoencoder to enable tactile sensing of 6D forces and torques for reactive grasp. As a result, we achieved multi-modal perception, including visual perception and tactile sensing, and soft, adaptive object grasping within a single vision-based soft finger design compatible with multi-fingered robotic grippers.
