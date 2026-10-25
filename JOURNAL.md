### Project Journal
# ResNet CNN | HM10000 Skin Lesion Detection
by: Keegan.H.K

## Day 1:
First day of the project started off by extracting the dataset from HuggingFace source locally on my machine in the `/data` folder which is excluded from git due to the file size. The dataset was split into its training and test datasets *(75 / 25 split)*. Due to the fact that the model used to process and identify whether a skin lesion / mole was malignant or benign from an image meant the model needed to be a `Convolutional Neural Network (CNN)` to breakdown the pixels of the image using a multi-layer network with the inputs being the image of the skin lesion and the output producing the probability whether it was malignant or benign.

Once the dataset was downloaded locally, I created a data verification script `quality.py` that runs through the images to check for any duplicate images that could contaminate the dataset. Furthermore the images are read and fingerprinted twice at about 250 images per second using:
- *MD5*: a fingerprint of the exact file bytes to catch byte-identical copies.
- *Perceptual Hash*: a fingerprint of how the image looks even after resizing, recompression or slight changes made to the original image.

The script removed 2 duplicate images, 28 holdout images and 187 lesion pairs were grouped as near duplicates.

With the dataset now finished processing and the removal of two duplicate images from the dataset, we could proceed with the initial training of the model. The model of choice was the pre-existing `ResNet18` (Residual Network), this model was the perfect choice due to it requiring a model to process images which this deep Convolutional neural network model provides. The only change made to the model was the output layer by the new output for the two results, either Malignant or benign.

The model was pretrained off ImageNet (transfer learning), which is why it scored AUC 0.83 before fine-tuning and as the samllest standered ResNet, it trains quickly on a laptop GPU chip like my M1 and keeps that knowledge when the training dataset is run through the model to adjust the weights appropriately. The learning rate is set to 0.001 for the Head and 0.0001 for the AdamW optimiser.

The test and holdout datasets will be kept for the final run once fine-tuning has reached a more accurate position.