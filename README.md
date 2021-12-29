# Survey on Convolutional Neural Networks models that Detects COVID-19 using Chest X-Ray Images

## [(Click here) for Whitepaper](docs/Team_1619_CS598_DLH_Project_Final_Report.pdf)

This [paper]((docs/Team_1619_CS598_DLH_Project_Final_Report.pdf))intends to provide an unbiased and neutral review of four existing models that detects COVID-19 from Chest X-Ray images using Convolutional Neural Networks(CNNs). This comparative survey aims to aid the researchers and experts in choosing the most relevant model to detect COVID-19 patients.

### How to run the code

Four CNN models have been implemented based on their original research papers. The executable code resides in Jupyter notebook files which need to be run from the Google Colab environment. Click on the **Open in Colab** button next to the each of the files below to open the file in Colab.

1.  The executable code resides in the following files.
    - _CoroNet.ipynb_      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cZtGDLNPNTOkY8sw34Uwj6CAcwj7xb9h?usp=sharing)
    - _DarkCovidNet.ipynb_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Bysmfwh7CNVUIB7M404hDK1wZmStPUNO?usp=sharing)
    - _EMCNet.ipynb_       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WT4EFr8HNkHr-Px43iL9NcnP0YNKyOf3?usp=sharing)
    - _Haque_et_al.ipynb_  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_bZekHnJFAki-5xqZ2IGDhskj9DhwEJn?usp=sharing)

2.  Once in Colab, the code needs to run on a GPU. From Colab, navigate to **Edit> Notebook Settings**. Select **GPU** from the *Hardware accelerator* dropdown

3.  The default settings would run the models for 100 epochs. If this or any other hyperparameters need to be tuned, the following section in the notebook can be modified.

    ```python
    # If True, the output will be stored in Google Drive and will be permanent. Otherwise it will be stored in the Colab workspace which is highly volatile
    save_cloud = False

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 100
    ```
   

4. The notebook can be executed in two ways
    - To execute the entire notebook in one shot, click **Runtime> Run all**
    - To execute one code block at a time, navigate through the code blocks in order and click on the black 'Play' button at the top left corner of each block.


5.  At the end of code execution, the prediction scores and metrics graphs can be seen at the last few slides of the notebook.

A video presentation is available [HERE](https://www.youtube.com/)

### How the code works

The various models use CNNs to scan Chect X-Rays and classifies whether the patient as COVID-19 infected or not.

This is how the code works at the high level

1. Install/import packages and set global variables
2. Download the X-Ray dataset from a shared Google Drive folder and unzip it in the Colab workspace
3. Split the dataset into Train, Validation and Training in the ratio 60:20:20
4. Do image pre-processing and load the train, validation and test dataloaders
5. Define the CNN model as presented in the paper
6. Perform training. For 100 epochs, perform the following steps
    - Train the model on the training dataset
    - Calculate training loss and accuracy
    - Validate the model on the validation dataset
    - Calculate validation loss and accuracy
    - If validation accuracy has improved, save the model
    - Save the checkpoint so that training can be resumed from that epoch if the training fails
   
   At the end of training, return the model with the best validation accuracy
7. Test the model on the testing dataset using the model that was trained
8. Calculate performance metrics and plot graphs

### Dependencies

- `python` &emsp;
- `torch` &emsp;
- `numpy` &emsp;
- `torchvision` &emsp;
- `gdown` &emsp;
- `splitfolders` &emsp;
- `matplotlib` &emsp;
- `google.colab` &emsp;
- `sklearn.metrics` &emsp;
- `timm` &emsp;
- `torchsummary` &emsp;
- `treepy` &emsp;







### References
1. [CoroNet: A deep neural network for detection and diagnosis of COVID-19 from chest x-ray images](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7274128/)
2. [A Deep Learning Approach to Detect COVID-19 Patients from Chest X-ray Images](https://www.researchgate.net/publication/344340531_A_Deep_Learning_Approach_to_Detect_COVID-19_Patients_from_Chest_X-ray_Images)
3. [Automated COVID-19 diagnosis from X-ray images using convolutional neural network and ensemble of machine learning classifiers](https://www.sciencedirect.com/science/article/pii/S2352914820306560)
4. [Automated detection of COVID-19 cases using deep neural networks with X-ray images](https://www.sciencedirect.com/science/article/abs/pii/S0010482520301621)
5. [ImageNet training in PyTorch](https://github.com/pytorch/examples/blob/537f6971872b839b36983ff40dafe688276fe6c3/imagenet/main.py)
6. [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
7. [Kaggle COVID-19 Radiography Dataset](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
8. [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)
9. [XRay CNN in PyTorch](https://www.kaggle.com/salvation23/xray-cnn-pytorch)
10. [Plot ROC curve](https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python)
11. [Plot Confusion Matrix](https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
12. [Xception paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)
13. [Xception for PyTorch](https://rwightman.github.io/pytorch-image-models/models/xception/)
14. [CIFAR10 tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) 
15. [Original CoroNet GitHub Repo](https://github.com/drkhan107/CoroNet)
16. [Original DarkCovidNet GitHub Repo](https://github.com/muhammedtalo/COVID-19/blob/master/)

