//
//  main.cpp
//  OpenCV_DT_Demo
//
//  Created by John Lunsford on 5/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "cv.h"
#include "ml.h"


// The neural network
CvDTree dtree;


// Read the training data and train the network.
void trainMachine()
{
    int i;
    //The number of training samples. 
    int train_sample_count = 1439;
    
    //The training data matrix. 
    float td[1439][12];
    
    //Read the training file
    FILE *fin;
    fin = fopen("data/winequality_train.csv", "r");
    
    //Create the matrices    
    //Input data samples. Matrix of order (train_sample_count x 11)
    CvMat* trainData = cvCreateMat(train_sample_count, 11, CV_32FC1);
    
    //Output data samples. Matrix of order (train_sample_count x 1)
    CvMat* trainClasses = cvCreateMat(train_sample_count, 1, CV_32FC1);
    
    
    //The matrix representation of our feature data types.
    CvMat* var_Types = cvCreateMat(12, 1, CV_8U );
    cvSet(var_Types, cvScalarAll(CV_VAR_ORDERED));
    cvSet2D(var_Types, 11, 0, cvScalarAll(CV_VAR_CATEGORICAL));
    
    CvMat trainData1, trainClasses1;
    
    cvGetRows(trainData, &trainData1, 0, train_sample_count);
    cvGetRows(trainClasses, &trainClasses1, 0, train_sample_count);
    cvGetRows(trainClasses, &trainClasses1, 0, train_sample_count);
    
    
    //Read and populate the samples.
    //fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality
    for (i=0;i<train_sample_count;i++)
        fscanf(fin,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
               &td[i][0],&td[i][1],&td[i][2],&td[i][3],&td[i][4],&td[i][5],&td[i][6],&td[i][7],&td[i][8],&td[i][9],&td[i][10],&td[i][11]);
    
    //we are done reading the file, so close it
    fclose(fin);
    
    //Assemble the ML training data.
    for (i=0; i<train_sample_count; i++)
    {
        //inputs
        for (int j = 0; j < 11; j++) 
            cvSetReal2D(&trainData1, i, j, td[i][j]);
    
        //Output
        cvSet1D(&trainClasses1, i, cvScalar(td[i][11]));
    }
    
    //Create our ANN.
    cout << "training...\n";
    //Train it with our data.
    dtree.train(trainData, CV_ROW_SAMPLE, trainClasses);
}

// Predict the output with the trained ANN given the two inputs.
void predict()
{
    int test_sample_count = 160;
    
    //The test data matrix. 
    float td[160][12];
    float _sample[11];
    CvMat sample = cvMat(1, 11, CV_32FC1, _sample);
    
    //Read the test file
    FILE *fin;
    fin = fopen("data/winequality_test.csv", "r");
    
    //fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality
    for (int i=0; i<test_sample_count; i++)
        fscanf(fin,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
               &td[i][0],&td[i][1],&td[i][2],&td[i][3],&td[i][4],&td[i][5],&td[i][6],&td[i][7],&td[i][8],&td[i][9],&td[i][10],&td[i][11]);
    
    fclose(fin);
    float errorTally = 0.0f;
    int errorCount = 0;
    float scoreTally = 0.0f;
    for (int i=0; i < test_sample_count; i++)
    {
        for (int j=0; j < 11; j++) {
            sample.data.fl[j] = td[i][j];
        }
        float actual = td[i][11];
        double predicted = dtree.predict(&sample, 0, false)->value;
       
        if (actual < predicted)
        {
            errorTally += (predicted - actual);
            errorCount++;
        }
        else if (predicted < actual)
        {
            errorTally += (actual - predicted);
            errorCount++;
        }
    
        scoreTally += actual;
        printf("predicted: %f, actual: %f\n", predicted, actual);
    }
    std::cout << "Error Count: " << errorCount << "\n";
    std::cout << "Error Percentage: " << ((float)errorCount / test_sample_count) * 100 << "%\n";
    std::cout << "Average Error: " << (errorTally / errorCount) << "\n";
}


int main (int argc, const char * argv[])
{
    std::cout << "Wine quality demo\n";
    
    // Train the neural network  with the samples
    trainMachine();
    
    // Now try predicting some values with the trained network
    predict();
    
    return 0;
}

