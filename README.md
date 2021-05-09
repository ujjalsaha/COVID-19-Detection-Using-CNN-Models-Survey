# Survey on Convolutional Neural Networks models that Detects COVID-19 using Chest X-Ray Images

This paper intends to provide an unbiased and neutral review of four existing models that detects COVID-19 from Chest X-Ray images using Convolutional Neural Networks(CNNs). This comparative survey aims to aid the researchers and experts in choosing the most relevant model to detect COVID-19 patients.

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
    save_cloud = True

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 100
    ```

   

4. The notebook can be executed in two ways
- To execute in one 
- by executing all the code blocks in order by clicking on the black 'Play' button at the top of each block.

5.  In the end, the prediction scores and metrics are

A video tutorial is available [HERE](https://www.youtube.com/)

### How the code works

This project uses CNNs


This is how the code works at the high level

 1. Step 1
 2. Step 2
3. 

### Dependencies

-   python
-    pytorch
-   os



### References
1. [Text](Link) by **Orhan G. Yalçın**

