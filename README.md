# ML-covid-project

The mini-project covers the application of the two algorithms (supervised and
unsupervised) in the context of covid19. The use of one or
the set of two datasets is optional. Note that the first is a
csv file containing the observations on all the samples and the
second contains chest scans

#choice of Algorithms

Choice of the algorithm - A step towards the solution
In fact, the first intuition that comes to mind when interacting with the
first dataset is to situate the context of the problem. We are in
interaction with a dataset containing nominal data and
quantitative. Therefore, we have the symptoms present on the
various observations to name but a few: cough,
fever, shortness of breath. So why not predict patient outcome
just from these symptoms.
This now amounts to being interested only in those said quantitative attributes
and the class corona-result which gives the positive or negative result from
of this.

#1) To be able to execute this project, you need to install:
pip install pandas numpy install scikit-learn matplotlib seaborn

#You can install the different packages by the conda environment


#the first knn.py for the supervised
#clustering.py for the unsupervised
