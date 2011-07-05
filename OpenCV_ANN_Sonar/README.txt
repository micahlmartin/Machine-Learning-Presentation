Detect Mines Demo

This demo uses data collected by bouncing sonar signals off of various objects (some of which are sea mines) to train an artificial neural network to predict whether an object is a mine or not.

The code is written in C++ and requires the OpenCV library. OpenCV can be found here: http://opencv.willowgarage.com/wiki/. It can also be obtained via MacPorts (which is the easiest way and what I would recommend). I developed the demo using XCode 4. But if you do not have XCode, the main.cpp file can easily be compiled with any other compiler. Just note that, the way the code is written now, it looks for the data files in a "Data" subfolder relative to the executable's filesystem location.