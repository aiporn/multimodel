# multimodel

This is a model that does multiple porn-related tasks at once. It regresses against hotspots, classifies categories and tags, and predicts view counts.

# Initial results

This section contains the initial results on a dump of 45K videos (a total of over 30GB).

Fitting the full objective, the model seems to bottom out at a loss of about 0.85. The loss arrives at this point pretty much right away and doesn't go down much afterwards. Virtually no overfitting is visible.

Fitting just the categorization objective, after 50K iterations with learning rate 1e-4, the training loss gets down to ~0.12 while the validation loss is around ~0.14. At this point, the training loss is still decreasing steadily. Thus, it would seem that the network is capable of learning to categorize videos, but it requires more data.

*More results pending*
