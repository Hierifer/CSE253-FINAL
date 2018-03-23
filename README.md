# Convolutional CTC for recognizing one sequence of math symbols or equation
This project combines convolutional network and CTC to recognize the sequence of math symbols or equation. It builds on Tensorflow for Tensorflow_CTCLoss. In the result, It reachs high accuracy of nearly 95% on the arbitary dataset created by our auto-generator. We begin with igormq's repository: https://github.com/igormq/ctc_tensorflow_example . And, modify the original model to adapt new dataset.  

## Authors
* **Changliang Cao** - *Initial work* - Email: chc506@eng.ucsd.edu
* **Teng Hu** - *Initial work* - Email: teh007@eng.ucsd.edu
* **Zijia Chen** - *Initial work* - Email: zic138@eng.ucsd.edu

## Getting Started
The project needs Tensorflow. Thus, your machine needs to install Tensorflow and cuda (if you want to run on your GPU). This repository only load a few image data from Kaggle --- https://www.kaggle.com/xainano/handwrittenmathsymbols. You supposes to download more data images on the data folders. For we only tested on "0-9,+,-,times,div,=", if you want to test more, please change the variable "alphabet" in "ctc_tensorflow_multidata_example.py".

## Running the demo
Run by "ctc_tensorflow_multidata_example.py"

## Built With
* [anaconda 3](https://anaconda.org/) - anaconda
* [Original CTC Model](https://github.com/igormq/ctc_tensorflow_example) - Arthor: igormq
* [Tensorflow](https://www.tensorflow.org/) - Tensorflow

## Acknowledgments
* Yue Xie --- Technic Support
* Zhuo Cheng --- Technic Support & Data Supplies

