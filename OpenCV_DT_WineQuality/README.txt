Wine Quality Demo

This demo uses data about the chemical composition of several wines plus a rating of each wine on a scale of 1-10 to train a decision tree to predict a wine's quality.

The code is written in C++ and requires the OpenCV library. OpenCV can be found here: http://opencv.willowgarage.com/wiki/. It can also be obtained via MacPorts (which is the easiest way and what I would recommend). I developed the demo using XCode 4. But if you do not have XCode, the main.cpp file can easily be compiled with any other compiler. Just note that, the way the code is written now, it looks for the data files in a "Data" subfolder relative to the executable's filesystem location.