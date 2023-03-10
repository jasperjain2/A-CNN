Hi Dr. J,

So I'm turning in some code that doesn't work at all. Hopefully it will work after this weekend.

I started off with a dataset of monkeys that all looked quite similar (~1300 photos in 10 classes), and had difficulty getting my model to classify those. I ran into issues with overtraining (accuracy values >.9 with validation accuracies sticking around .7). I've spent a good deal of time trying to combat this with dropout and data augmentation, though that dataset might just have been too small. It also had some issues with watermarks and bad photos. 

I recently downloaded a dataset of a bunch of different types of animals, which should hopefully be easier to distinguish. Even with this dataset, my accuracies seem to plateau a low values pretty quickly, so I think my program is running into some strange local minima. I've been playing around with some different optimizers, and plugging in bogus learning rates has been instructional.

Currently, I'm working on displaying the first few layers of the CNN. I've stolen your code to do that and am figuring that out at the moment. For now, those are commented out so that the model can actually be run.

Thanks!
Jasper

**update: I've broken something. Or I just need to be patient and train for 9 hours again. Perhaps some parameter is way off?
