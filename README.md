# SeeThruFinger: See and Grasp Anything with a Soft Touch
![fig-PaperOverview](./tools/fig-PaperOverview.png)

We present SeeThruFinger, a soft robotic finger with an in-finger vision for multi-modal perception, including visual perception and tactile sensing, for geometrically adaptive and real-time reactive grasping. 

Multi-modal perception of intrinsic and extrinsic interactions is critical in building intelligent robots that learn. Instead of adding various sensors for different modalities, a preferred solution is to integrate them into one elegant and coherent design, which is a challenging task. This study leverages the Soft Polyhedral Network design as a robotic finger, capable of omni-directional adaptation with an unobstructed view of the finger's spatial deformation from the inside. 

By embedding a miniature camera underneath, we achieve the visual perception of the external environment by inpainting the finger mask using E2FGV, which can be used for object detection in the downstream tasks for grasping. After contacting the objects, we use real-time object segmentation algorithms, such as XMem, to track the soft finger's spatial deformations. We also learned a Supervised Variational Autoencoder to enable tactile sensing of 6D forces and torques for reactive grasp. As a result, we achieved multi-modal perception, including visual perception and tactile sensing, and soft, adaptive object grasping within a single vision-based soft finger design compatible with multi-fingered robotic grippers.


https://github.com/ancorasir/SeeThruFinger/assets/12664844/c33f6374-23fe-4481-a13b-1c23b014c4b8

