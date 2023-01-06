Machine Learning Notebooks, 3rd edition
=================================

This project aims at teaching you the fundamentals of Machine Learning in
python. It contains the example code and solutions to the exercises in the third edition of my O'Reilly book [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow (3rd edition)](https://homl.info/er3):

<a href="https://homl.info/er3"><img src="https://learning.oreilly.com/library/cover/9781098125967/300w/" title="book" width="150" border="0" /></a>

**Note**: If you are looking for the second edition notebooks, check out [ageron/handson-ml2](https://github.com/ageron/handson-ml2). For the first edition, see [ageron/handson-ml](https://github.com/ageron/handson-ml).

## Quick Start

### Want to play with these notebooks online without having to install anything?

* <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> (recommended)

⚠ _Colab provides a temporary environment: anything you do will be deleted after a while, so make sure you download any data you care about._

<details>

Other services may work as well, but I have not fully tested them:

* <a href="https://homl.info/kaggle3/"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" /></a>

* <a href="https://mybinder.org/v2/gh/ageron/handson-ml3/HEAD?filepath=%2Findex.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch binder" /></a>

* <a href="https://homl.info/deepnote3/"><img src="https://deepnote.com/buttons/launch-in-deepnote-small.svg" alt="Launch in Deepnote" /></a>

</details>

### Just want to quickly look at some notebooks, without executing any code?

* <a href="https://nbviewer.jupyter.org/github/ageron/handson-ml3/blob/main/index.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Render nbviewer" /></a>

* [github.com's notebook viewer](https://github.com/ageron/handson-ml3/blob/main/index.ipynb) also works but it's not ideal: it's slower, the math equations are not always displayed correctly, and large notebooks often fail to open.

### Want to run this project using a Docker image?
Read the [Docker instructions](https://github.com/ageron/handson-ml3/tree/main/docker).

### Want to install this project on your own machine?

Start by installing [Anaconda](https://www.anaconda.com/products/distribution) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), [git](https://git-scm.com/downloads), and if you have a TensorFlow-compatible GPU, install the [GPU driver](https://www.nvidia.com/Download/index.aspx), as well as the appropriate version of CUDA and cuDNN (see TensorFlow's documentation for more details).

Next, clone this project by opening a terminal and typing the following commands (do not type the first `$` signs on each line, they just indicate that these are terminal commands):

    $ git clone https://github.com/ageron/handson-ml3.git
    $ cd handson-ml3

Next, run the following commands:

    $ conda env create -f environment.yml
    $ conda activate homl3
    $ python -m ipykernel install --user --name=python3

Finally, start Jupyter:

    $ jupyter notebook

If you need further instructions, read the [detailed installation instructions](INSTALL.md).

# FAQ

**Which Python version should I use?**

I recommend Python 3.10. If you follow the installation instructions above, that's the version you will get. Any version ≥3.7 should work as well.

**I'm getting an error when I call `load_housing_data()`**

If you're getting an HTTP error, make sure you're running the exact same code as in the notebook (copy/paste it if needed). If the problem persists, please check your network configuration. If it's an SSL error, see the next question.

**I'm getting an SSL error on MacOSX**

You probably need to install the SSL certificates (see this [StackOverflow question](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)). If you downloaded Python from the official website, then run `/Applications/Python\ 3.10/Install\ Certificates.command` in a terminal (change `3.10` to whatever version you installed). If you installed Python using MacPorts, run `sudo port install curl-ca-bundle` in a terminal.

**I've installed this project locally. How do I update it to the latest version?**

See [INSTALL.md](INSTALL.md)

**How do I update my Python libraries to the latest versions, when using Anaconda?**

See [INSTALL.md](INSTALL.md)

## Contributors
I would like to thank everyone [who contributed to this project](https://github.com/ageron/handson-ml3/graphs/contributors), either by providing useful feedback, filing issues or submitting Pull Requests. Special thanks go to Haesun Park and Ian Beauregard who reviewed every notebook and submitted many PRs, including help on some of the exercise solutions. Thanks as well to Steven Bunkley and Ziembla who created the `docker` directory, and to github user SuperYorio who helped on some exercise solutions. Thanks a lot to Victor Khaustov who submitted plenty of excellent PRs, fixing many errors. And lastly, thanks to Google ML Developer Programs team who supported this work by providing Google Cloud Credit.
