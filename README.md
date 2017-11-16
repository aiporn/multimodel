# multimodel

This is a model that does multiple porn-related tasks at once. It regresses against hotspots, classifies categories and tags, and predicts view counts.

# Initial results

This section contains the initial results on a dump of 45K videos (a total of over 30GB).

Fitting the **full objective**, the model doesn't learn much. It seems to bottom out at a loss of about 0.85. The loss arrives at this point pretty much right away and doesn't go down much afterwards. Virtually no overfitting is visible.

Fitting just the **categorization objective**, the model definitely learns something but overfits quickly. After 50K iterations with learning rate 1e-4, the training loss gets down to ~0.12 while the validation loss is around ~0.14. At this point, the training loss is still decreasing steadily. Thus, it would seem that the network is capable of learning to categorize videos, but it requires more data.

Fitting just the **hotspot objective**, the model doesn't learn much. The loss bottoms out at around 0.136, but it is extremely noisy. The training loss is only a tiny bit lower than the validation loss, even after 120K iterations.

Fitting just the **popularity objective**, the model overfits a bit but doesn't learn much overall. After ~100K iterations, the training loss is ~0.55 and the validation loss is ~0.56.

Fitting just the **like loss** portion of the **popularity objective**, the model overfits a bit. This sub-loss seems to dominate the popularity objective. The training loss gets down to ~0.530, while the validation loss is ~0.547.

Fitting just the **view loss** portion of the **popularity objective**, the model overfits by a great margin. The training loss gets to 0.009, while the test loss remains at 0.021.

*More results pending*
