# ScatterPlotAnalyzer (SPA) 

Data extraction of Bar charts and its' variants using tensor field computation | [Paper](https://www.iiitb.ac.in/GVCL/pubs/2021_DadhichDaggubatiSreevalsanNair_ICCS_preprint.pdf) |

## Text Detection and Recognition Module

This module performs text detection and recognition on chart Image. We use a deep-learning-based OCR, namely Character Region Awareness for Text Detection, CRAFT | [Paper](https://arxiv.org/abs/1904.01941) | succeeded by a scene text recognition framework, STR | [Paper](https://arxiv.org/abs/1904.01906) |
 
### To run the code

Things to be taken care before runing the code:
1. Fetch CRAFT text detection code from [CRAFT_TextDetector](https://github.com/GVCL/Tensor-field-framework-for-chart-analysis/tree/master/Chart-Analyzer), and place dir at the following path ```ScatterPlotAnalyzer_SPA/CRAFT_TextDetector```
2. Download the pretrained model [craft_mlt_25k.pth](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view), and place model at the following path ```ScatterPlotAnalyzer_SPA/CRAFT_TextDetector/craft_mlt_25k.pth```
3. Fetch STR text recognition code from [Deep_TextRecognition](https://github.com/GVCL/Tensor-field-framework-for-chart-analysis/tree/master/Chart-Analyzer), and place dir at the following path ```ScatterPlotAnalyzer_SPA/Deep_TextRecognition```
4. Download the pretrained model [TPS-ResNet-BiLSTM-Attn.pth](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW), tand place model at the following path ```ScatterPlotAnalyzer_SPA/Deep_TextRecognition/TPS-ResNet-BiLSTM-Attn.pth```
5. The code is developed and tested on Python 3.6 you can also find attached requirements.txt to avoid errors due to compatibility issues
6. Annotate the chart image and generate it's ```image_name.xml``` file using LabelImg annotation tool | [Tool](https://github.com/tzutalin/labelImg) | [Demo Video](https://www.youtube.com/watch?v=t3rlG_v8sMs) |. Make sure image file and its .xml file are in same directory
7. Finally you can run the  ```Image_uploader.py``` file and upload desired chart image file.

This wil generate the folowing csv files:
1. Image_RGB.csv: contains RGB values along with xy-coordinates.
2. structure_tensor.csv: contains structure tensor matrix, eigen values, eigen vectors and cl-cp values.
3. tensor_vote_matrix.csv: contains tensor voting matrix, eigen values, eigen vectors and cl-cp values.

For visualize the csv files using Tensor field visualization | [Code](https://github.com/GVCL/Tensor-field-framework-for-chart-analysis/tree/master/Chart-Analyzer/Visualizer) |:
```
1. cd Visualizer
2. python csv_uploader.py
3. upload required csv file for structure tensor/tensor vote visualization.
```
