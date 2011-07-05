//
//  main.cpp
//  OpenCV_ANN_Demo
//
//  Created by John Lunsford on 5/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "cv.h"
#include "ml.h"


// The neural network
CvANN_MLP ann;


// Read the training data and train the network.
void trainMachine()
{
    int i;
    //The number of training samples. 
    int train_sample_count = 130;
    
    //The training data matrix. 
    float td[130][61];
    
    //Read the training file
    FILE *fin;
    fin = fopen("data/sonar_train.csv", "r");
    
    //Create the matrices    
    //Input data samples. Matrix of order (train_sample_count x 60)
    CvMat* trainData = cvCreateMat(train_sample_count, 60, CV_32FC1);
    
    //Output data samples. Matrix of order (train_sample_count x 1)
    CvMat* trainClasses = cvCreateMat(train_sample_count, 1, CV_32FC1);
    
    //The weight of each training data sample. We'll later set all to equal weights.
    CvMat* sampleWts = cvCreateMat(train_sample_count, 1, CV_32FC1);
    
    //The matrix representation of our ANN. We'll have four layers.
    CvMat* neuralLayers = cvCreateMat(4, 1, CV_32SC1);
    
    //Setting the number of neurons on each layer of the ANN
    /* 
     We have in Layer 1: 60 neurons (60 inputs)
     Layer 2: 150 neurons (hidden layer)
     Layer 3: 225 neurons (hidden layer)
     Layer 4: 1 neurons (1 output)
     */
    cvSet1D(neuralLayers, 0, cvScalar(60));
    cvSet1D(neuralLayers, 1, cvScalar(150));
    cvSet1D(neuralLayers, 2, cvScalar(225));
    cvSet1D(neuralLayers, 3, cvScalar(1));
    
    //Read and populate the samples.
    for (i=0;i<train_sample_count;i++)
        fscanf(fin,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
               &td[i][0],&td[i][1],&td[i][2],&td[i][3],&td[i][4],&td[i][5],&td[i][6],&td[i][7],&td[i][8],&td[i][9],&td[i][10],&td[i][11],&td[i][12],&td[i][13],&td[i][14],&td[i][15],&td[i][16],&td[i][17],&td[i][18],&td[i][19],&td[i][20],&td[i][21],&td[i][22],&td[i][23],&td[i][24],&td[i][25],&td[i][26],&td[i][27],&td[i][28],&td[i][29],&td[i][30],&td[i][31],&td[i][32],&td[i][33],&td[i][34],&td[i][35],&td[i][36],&td[i][37],&td[i][38],&td[i][39],&td[i][40],&td[i][41],&td[i][42],&td[i][43],&td[i][44],&td[i][45],&td[i][46],&td[i][47],&td[i][48],&td[i][49],&td[i][50],&td[i][51],&td[i][52],&td[i][53],&td[i][54],&td[i][55],&td[i][56],&td[i][57],&td[i][58],&td[i][59],&td[i][60]);
    
    //we are done reading the file, so close it
    fclose(fin);
    
    //Assemble the ML training data.
    for (i=0; i<train_sample_count; i++)
    {
        //inputs
        for (int j = 0; j < 60; j++) 
            cvSetReal2D(trainData, i, j, td[i][j]);
    
        //Output
        cvSet1D(trainClasses, i, cvScalar(td[i][60]));
        //Weight (setting everything to 1)
        cvSet1D(sampleWts, i, cvScalar(1));
    }
    
    //Create our ANN.
    ann.create(neuralLayers);
    cout << "training...\n";
    //Train it with our data.
    ann.train(
        trainData,
        trainClasses,
        sampleWts,
        0,
        CvANN_MLP_TrainParams(
            cvTermCriteria(
                CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                100000,
                0.000001),
            CvANN_MLP_TrainParams::BACKPROP,
            0.01,
            0.05));
}

// Predict the output with the trained ANN given the two inputs.
void predict()
{
    int test_sample_count = 78;
    
    //The test data matrix. 
    float td[78][61];
    float _sample[60];
    CvMat sample = cvMat(1, 60, CV_32FC1, _sample);
    float _predout[1];
    CvMat predout = cvMat(1, 1, CV_32FC1, _predout);
    
    //Read the test file
    FILE *fin;
    fin = fopen("data/sonar_test.csv", "r");
    
    for (int i=0; i<test_sample_count; i++)
        fscanf(fin,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
               &td[i][0],&td[i][1],&td[i][2],&td[i][3],&td[i][4],&td[i][5],&td[i][6],&td[i][7],&td[i][8],&td[i][9],&td[i][10],&td[i][11],&td[i][12],&td[i][13],&td[i][14],&td[i][15],&td[i][16],&td[i][17],&td[i][18],&td[i][19],&td[i][20],&td[i][21],&td[i][22],&td[i][23],&td[i][24],&td[i][25],&td[i][26],&td[i][27],&td[i][28],&td[i][29],&td[i][30],&td[i][31],&td[i][32],&td[i][33],&td[i][34],&td[i][35],&td[i][36],&td[i][37],&td[i][38],&td[i][39],&td[i][40],&td[i][41],&td[i][42],&td[i][43],&td[i][44],&td[i][45],&td[i][46],&td[i][47],&td[i][48],&td[i][49],&td[i][50],&td[i][51],&td[i][52],&td[i][53],&td[i][54],&td[i][55],&td[i][56],&td[i][57],&td[i][58],&td[i][59],&td[i][60]);
    
    fclose(fin);
    int fnCount = 0;
    int fpCount = 0;
    for (int i=0; i < test_sample_count; i++)
    {
        for (int j=0; j < 60; j++) {
            sample.data.fl[j] = td[i][j];
        }
        float actual = td[i][60];
        
        ann.predict(&sample, &predout);
        
        float predicted = predout.data.fl[0];
        if (actual == 1.0f && predicted < 0.0f) 
        {
            fnCount++;
            std::cout << "BOOM! ";
        }
        else if (actual == -1.0f && predicted > 0.0f) 
        {
            fpCount++;
        }
    
        printf("predicted: %f, actual: %f\n", predicted, actual);
    }
    
    std::cout << "False Negative %: " << ((float)fnCount / test_sample_count)*100 << "%\n";
    std::cout << "False Positive %: " << ((float)fpCount / test_sample_count)*100 << "%\n";
    std::cout << "Total Misses: " << ((float)(fpCount+fnCount) / test_sample_count)*100 << "%\n";
}


int main (int argc, const char * argv[])
{
    std::cout << "Sonar mine demo\n";
    
    // Train the neural network  with the samples
    trainMachine();
    
    // Now try predicting some values with the trained network
    predict();
    
    return 0;
}

