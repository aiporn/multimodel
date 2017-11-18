# multimodel

This is a model that does multiple porn-related tasks at once. It regresses against hotspots, classifies categories and tags, and predicts view counts.

# Initial results

This section contains the initial results on a dump of 45K videos (a total of over 30GB). Different objectives were trained for different amounts of time, but all were given time to learn *something*:

 * **full objective:** training=0.85
   * **categorization objective:** training=0.12 validation=0.14 (still improving on training set)
   * **hotspot objective** training=0.136 (quite noisy)
   * **popularity objective:** training=0.55  validation=0.56
     * **like loss:** training=0.530 validation=0.547
     * **view loss:** training=0.009 validation=0.021

Conclusions: There is not enough data, and/or there needs to be more data augmentation and regularization. Also, the view loss should be weighted more heavily.

# After data augmentation

 * **categorization objective:** with lr=1e-4 and iters=170K, training=0.142200 validation=0.159802 (still improving)
