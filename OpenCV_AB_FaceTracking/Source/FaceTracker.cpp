
#include "cv.h"
#include "highgui.h"
#include <cassert>
#include <iostream>


const char  * WINDOW_NAME  = "Face Tracker";
const signed long CASCADE_NAME_LEN = 2048;
      char    CASCADE_NAME[CASCADE_NAME_LEN] = "data/haarcascade_frontalface_alt.xml";

using namespace std;

int main (int argc, char * const argv[]) 
{
    const int scale = 2;
        
    // create all necessary instances
    cvNamedWindow (WINDOW_NAME, CV_WINDOW_AUTOSIZE);
    CvCapture * camera = cvCreateCameraCapture (CV_CAP_ANY);
    CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*) cvLoad (CASCADE_NAME, 0, 0, 0);
    CvMemStorage* storage = cvCreateMemStorage(0);
    assert (storage);

    if (! camera)
        abort ();

    if (! cascade)
        abort ();

    // get an initial frame and duplicate it for later work
    IplImage *  current_frame = cvQueryFrame (camera);
    //Create a 3-channel buffer the same size as our capture for drawing
    IplImage *  draw_image    = cvCreateImage(cvSize (current_frame->width, current_frame->height), IPL_DEPTH_8U, 3);
    //Create a single channel buffer the same size as the capture. (This is an intermediate step)
    IplImage *  gray_image    = cvCreateImage(cvSize (current_frame->width, current_frame->height), IPL_DEPTH_8U, 1);
    //Create a smaller buffer that will contain a scaled-down, grayscale version of the capture
    //this is the buffer we will use for face detection (using a scaled-down single-channel buffer
    //boosts performance)
    IplImage *  small_image   = cvCreateImage(cvSize (current_frame->width / scale, current_frame->height / scale), IPL_DEPTH_8U, 1);
    
    // Loop while we are still capturing
    while ((current_frame = cvQueryFrame (camera)))
    {
        // convert to gray-scale ...
        cvCvtColor (current_frame, gray_image, CV_BGR2GRAY);
        // ... and scale down
        cvResize (gray_image, small_image, CV_INTER_LINEAR);
        
        // detect faces
        CvSeq* faces = cvHaarDetectObjects (small_image, cascade, storage,
                                            1.1, 2, CV_HAAR_DO_CANNY_PRUNING,
                                            cvSize (30, 30));
        
        // Rather than spend a bunch of cycles copying the capture buffer's
        //contents into the drawing buffer, we will just swap their
        //references
        cvFlip (current_frame, draw_image, 1);
        //draw a green circle around each face we found in our
        //capture image
        for (int i = 0; i < (faces ? faces->total : 0); i++)
        {
            CvRect* r = (CvRect*) cvGetSeqElem (faces, i);
            CvPoint center;
            int radius;
            center.x = cvRound((small_image->width - r->width*0.5 - r->x) *scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            cvCircle (draw_image, center, radius, CV_RGB(0,255,0), 1, 8, 0 );
        }
        
        // Draw our drawing buffer's contents to the window
        cvShowImage (WINDOW_NAME, draw_image);
        
        // wait a tenth of a second for keypress and window drawing
        int key = cvWaitKey (100);
        if (key == 'q' || key == 'Q')
            break;
    }
    
    // be nice and return no error
    return 0;
}
