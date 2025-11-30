# xurdif
a compact diffusion model trained with small, private datasets

This repo is essentially an updated and compact version of my urdiffusion (see https://github.com/htoyryla/urdiffusion)

Main changes: 

* Urdiffusion supported a variety of experimental Unet architectures, which has now been replaced with a new lightweight architecture with attention
* support for x0 prediction
* edge aware loss added

With these changes I have found that, using a coherent dataset of a few hundred images, a new model can be in as little as a few hours, even if for better quality 12 - 24 hours are recommended. Anyhow, already during the first hour or two it is possible so see what the model is learning.

