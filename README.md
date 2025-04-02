# GmClass
 A multimodal classifier with a force-text pair from robot-GM interaction.
 
This repository contains the source code and dataset ***GM10-ts*** and ***GM10-ts-Plus*** of the paper "A Joint Learning of Force Feedback of Robotic Manipulation and Textual Cues for Granular Material Classification," which is under review for RA-L.

* Project: https://sites.google.com/view/gmwork2/ftlearning

## News
A comprehensive dataset ***GM10-ts-Plus*** (10GMs, 27,000 data points) has been uploaded.

## Test Environment
* Ubuntu 20.04 or Windows 11

## Prerequisites 
* Python 3.9.15 (or above)
* Pytorch 1.13.1
* cuda 11.7
* [CLIP 1.0](https://github.com/openai/CLIP)

## Format of Data Point
In each data point (CSV file) in the dataset, each row refers to a raking experiment. 

If defining each row has M columns, then the 1-st column refers to the GM id.  

* In ***GM10-ts***:

From the 2-nd column to the (1+(M-1)/2)-th column, they are time series.

From the (2+(M-1)/2)-th column to the end, they are force series.


* In ***GM10-ts-Plus***:

From the 2-nd column to the 4-th column, they are (a, v, d) values.

From the 5-th column to the (5+(M-4)/2)-th column, they are time series.

From the (6+(M-4)/2)-th column to the end, they are force series.

## Enquiry:
Any problem, feedback, issue, or bug-finding is welcome as an open-source library.

Contact: Benji Z. Zhang (zzqing@connect.hku.hk)
