
======
From Fourier to Koopman: Spectral methods for long-term forecasts
======


Fourier and Koopman constitute spectral algorithms for learning linear and non-linear oscillators from data respectively.
Both algorithms solve a global optimization problem in frequency domain and allow for modeling of systems of any dimensionality.

The following implementation is in Python. This code accompanies the following `paper <https://arxiv.org/abs/xxx.xxxx>`_. 


-----------------
How to use Fourier
-----------------

Fourier fits a linear oscillator to data. The number of frequencies *k* that the signal is assumed to exhibit and a learning rate needs to be specified. It is recommended to whiten the signal (zero-mean and unit-variance).

To learn the oscillator from data, do:

.. code:: python

    from fourier_koopman import fourier
    import numpy as np

    x = np.sin(2*np.pi/24*np.arange(5000)) + np.sin(2*np.pi/33*np.arange(5000))

    f = fourier(k=2)
    f.fit(x[:3500], iterations = 1000)



To perform forecasting, do:

.. code:: python

    x_hat = f.predict(5000)

-----------------
How to use Koopman
-----------------

Because of the Running the Koopman algorithm is more involved and requires writing your own _model\_object_

.. code:: python

    from fourier_koopman import koopman, model_object
    import numpy as np

    x = np.sin(2*np.pi/24*np.arange(5000)) + np.sin(2*np.pi/33*np.arange(5000))

    f = fourier(k=2)
    f.fit(x[:3500], iterations = 1000)



To perform forecasting, do:

.. code:: python

    x_hat = f.predict(5000)


--------
Examples
--------

The following are some of the results on real-world datasets. The values of nearest-neighbor accuracy and global score are shown as a pair (NN, GS) on top of each figure. For more results, please refer to our `paper <https://arxiv.org/abs/1910.00204>`_.

USPS Handwritten Digits (*n = 11,000, d = 256*)

.. image:: results/usps.png
    :alt: Visualizations of the USPS dataset

20 News Groups (*n = 18,846, d = 100*)

.. image:: results/news20.png
    :alt: Visualizations of the 20 News Groups dataset

Tabula Muris (*n = 53,760, d = 23,433*)

.. image:: results/tabula.png
    :alt: Visualizations of the Tabula Muris Mouse Tissues dataset

MNIST Handwritten Digits (*n = 70,000, d = 784*)

.. image:: results/mnist.png
    :alt: Visualizations of the MNIST dataset

Fashion MNIST (*n = 70,000, d = 784*)

.. image:: results/fmnist.png
    :alt: Visualizations of the  Fashion MNIST dataset
    
TV News (*n = 129,685, d = 100*)

.. image:: results/tvnews.png
    :alt: Visualizations of the  TV News dataset


Runtime of t-SNE, LargeVis, UMAP, and TriMap in the hh:mm:ss format on a single machine with 2.6 GHz Intel Core i5 CPU and 16 GB of memory is given in the following table. We limit the runtime of each method to 12 hours. Also, UMAP runs out of memory on datasets larger than ~4M points.

.. image:: results/runtime.png
    :alt: Runtime of TriMap compared to other methods




------------------------
Support and Contribution
------------------------

This implementation is still a work in progress. Any comments/suggestions/bug-reports
are highly appreciated. Please feel free contact me at: eamid@ucsc.edu. If you would 
like to contribute to the code, please `fork the project <https://github.com/eamid/trimap/issues#fork-destination-box>`_
and send me a pull request.


--------
Citation
--------

If you use TriMap in your publications, please cite our current reference on arXiv:

::

   @article{2019TRIMAP,
        author = {{Amid}, E. and {Warmuth}, M. K.},
        title = "{TriMap: Large-scale Dimensionality Reduction Using Triplets}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1910.00204},
        year = 2019,
   }


-------
License
-------

Please see the LICENSE file.
