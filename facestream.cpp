#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){

    string model_path = "resources/haarcascade_frontalface_default.xml";
    string tie_path = "resources/bowtie.png";
    string hat_path = "resources/tophat.png";

    Mat3b src_img, mod_img;
    Mat4b tie = imread(tie_path, IMREAD_UNCHANGED);
    Mat4b hat = imread(hat_path, IMREAD_UNCHANGED);

    CascadeClassifier faceCascade;
    vector<Rect> faces;
    VideoCapture cap(0);

    // Hardcode resize of tie and hat for now
    int new_w = 150 * 1.2;
    int tie_h = (int)(tie.rows * new_w / tie.cols);
    int hat_h = (int)(hat.rows * new_w / hat.cols);
    resize(tie, tie, Size(new_w, tie_h));
    resize(hat, hat, Size(new_w, hat_h));
    
    faceCascade.load(model_path);

    if (faceCascade.empty()){      
            cout << "XML file not loaded" << endl;    
        }
    
    while (true){
        cap.read(src_img);

        faceCascade.detectMultiScale(src_img, faces, 1.1, 10);

        if (faces.size() > 0){

            int x, y; // x / y value of top left corner of picture to add
            Mat3b roi; 

            // Bowtie
            x = (int)(faces[0].br().x - (faces[0].width + tie.cols) / 2);
            y = faces[0].br().y + 20;

            if ((y < (src_img.rows - tie.rows - 20)) && (x > 0)){ // only show the picture, if all edges are in the frame
                roi = src_img(Rect(x,y,tie.cols, tie.rows));

                for (int r = 0; r < roi.rows; r++){
                    for (int c = 0; c < roi.cols; c++){
                        const Vec4b& vf = tie(r,c);
                        if (vf[3] > 100){
                            Vec3b& vb = roi(r, c);
                            vb[0] = vf[0];
                            vb[1] = vf[1];
                            vb[2] = vf[2];
                        }
                    }
                }
            }

            //Hat
            x = (int)(faces[0].tl().x + (faces[0].width - hat.cols) / 2 + 20); 
            y = (int)(faces[0].tl().y - hat.rows + 30);

            if ((y > 0) && (x < src_img.cols - hat.cols - 30)){ // only show the picture, if all edges are in the frame
                roi = src_img(Rect(x, y, hat.cols, hat.rows));

                for (int r = 0; r < roi.rows; r++){
                    for (int c = 0; c < roi.cols; c++){
                        const Vec4b& vf = hat(r,c);
                        if (vf[3] > 100){
                            Vec3b& vb = roi(r, c);
                            vb[0] = vf[0];
                            vb[1] = vf[1];
                            vb[2] = vf[2];
                        }
                    }
                }
            }
        }

        imshow("Faces", src_img);
        char c = (char)waitKey(25);
        if (c==27){
            break;
        }

    }

    return 0;
}