/**** Author Nizar Ouarti ****/
/******** Capture3D *********/

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
//#include <rs_types.h>
#include <librealsense2/hpp/rs_internal.hpp>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>

#include <opencv4/opencv2/imgproc/imgproc.hpp>

//#include <opencv2/calib3d.hpp>//rodriguez



#include <pcl/io/pcd_io.h> //write pcd
#include <pcl/io/ply_io.h> //write ply




#include <stdio.h>

using namespace std;
using namespace cv;

//config variables perhaps put it in a file
int width = 640;//1280; //480
int height = 480;//720; //270
int fps = 30;
double baseline=0.049963444;// recupere grace a l'outil de visu, realsense-viewer

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void visu(cv::Mat I, double scale, bool waitk){


	I.convertTo(I,CV_8UC1,scale);
    // Apply the colormap:
    Mat Ic;
    applyColorMap(I, Ic, COLORMAP_JET);

    cv::imshow("View",Ic);
    if (waitk) cv::waitKey(0);

}



void writeImage(Mat matIm, char* str, std::string dir_time, string id){

    char namedate[50];
    /*std::time_t time_now = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_now), "%y-%m-%d_%OH-%OM-%OS");
    std::string timing = ss.str();
    */
    sprintf(namedate,"%s/%s%s.png",dir_time.c_str(),str,id.c_str());

    printf("%s",namedate);


    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);//9 maximum compression
    
    try {
        imwrite(namedate, matIm, compression_params);
        fprintf(stdout, "Saved PNG file %s.\n",str);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());

    }
    
}



void intrinsics_extrinsics( rs2::pipeline_profile selection, rs2_stream CAM_TYPE,int cam_number){

    //rs2_intrinsics 
    //rs2_intrinsics intr;
    //rs2_extrinsics e;
    rs2::stream_profile stream;
    //cam_number utile uniquement avec les ca,eras infra-rouge
    if (CAM_TYPE==RS2_STREAM_INFRARED){
        stream = selection.get_stream(CAM_TYPE, cam_number);//1 == ir left | 2==ir right 
        
    }
    else{ 
        stream = selection.get_stream(CAM_TYPE);
    }

    auto stream_intr = stream.as<rs2::video_stream_profile>();
    rs2_intrinsics intr=stream_intr.get_intrinsics();
    auto princ_point=std::make_pair(intr.ppx,intr.ppy);
    auto focalexy=std::make_pair(intr.fx,intr.fy);
    rs2_distortion model=intr.model;

    cout << "Principal point : "<< intr.ppx << "  " <<intr.ppy<<  ", Focale x et y : "<< intr.fx << "  "<<intr.fy <<endl;

    cout << "Model: "<< model<<endl;    

    //Extrinsics

    //baseline
    //rs2_extrinsics e = stream.get_extrinsics_to(stream);
    rs2_extrinsics e = stream.get_extrinsics_to(stream);
    double baseline = e.translation[0];//baseline en metre
    baseline =-baseline ;// attention baseline negative!!!! 

    cout << baseline << baseline <<endl;

    //full extrinsics
    Mat transl(Size(1,3),CV_64F,cv::Scalar::all(0));
    transl.at<double>(0) = e.translation[0]+0.005;//translation entre irl et color selon x
    transl.at<double>(1) = e.translation[1];//translation entre irl et color selon y
    transl.at<double>(2) = e.translation[2];//translation entre irl et color selon z

    Mat ROT(Size(3,3),CV_64F,cv::Scalar::all(0));
    for (int i=0;i<9;i++)
        ROT.at<double>(i)= e.rotation[i];

    ROT=ROT.t();//je dois transposer car at() est row first

    cout << "Rotation" << ROT << "| Translation" <<transl << endl;



}



using pcl_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;

pcl_ptr points_to_pcl(const rs2::points& points)
{
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    auto ptr2 = points.get_texture_coordinates();
    // ptr2 pas utilisé ensuite ici il faut faire un lien avec la matrice rgb de couleur
    // p.r= dMat_color[ptr->y,ptr->x].r // cela dépend comment la couleur est stockée
    // Vec3b bgrPixel = dMat_color.at<Vec3b>(i, j);// acces indirect
    /*
    Direct ACCESS
uint8_t* pixelPtr = (uint8_t*)dMat_color.data;
int cn = dMat_color.channels();
Scalar_<uint8_t> bgrPixel;

for(int i = 0; i < dMat_color.rows; i++)
{
    for(int j = 0; j < dMat_color.cols; j++)
    {
        bgrPixel.val[0] = pixelPtr[i*foo.cols*cn + j*cn + 0]; // B
        bgrPixel.val[1] = pixelPtr[i*foo.cols*cn + j*cn + 1]; // G
        bgrPixel.val[2] = pixelPtr[i*foo.cols*cn + j*cn + 2]; // R

        // do something with BGR values...
    }
}
*/
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        //p.r = ptr->r;
        //p.g = ptr->g;
        //p.b = ptr->b;
        ptr++;
    }

    return cloud;
}




int main(){

    

    rs2::config config;

    //some options here
    //http://docs.ros.org/kinetic/api/librealsense2/html/rs__sensor_8h.html#a01b4027af33139de861408872dd11b93
    config.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);// on peut faire du Y16 je pense, plus precis -> il faudra change CV_8UC1 en CV_16UC1
    config.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
    config.enable_stream(RS2_STREAM_COLOR, width, height,   RS2_FORMAT_RGB8, fps);//1280 720
    config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);


    // start pipeline
    rs2::pipeline pipeline;
    rs2::pipeline_profile pipeline_profile = pipeline.start(config);

    cout<< "press 's' to record and 'q' to quit the program. Many recordings are possible." <<endl;



    //current time when the program is launched

    char namedate[50];
    std::time_t time_now = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_now), "%y-%m-%d_%OH-%OM-%OS");
    std::string timing = ss.str();
    //name of directory
    std::string dir_time="data_"+timing;

    int cmpt=0;//incremente si on presse 's'

    // end current time

    //alignment not in the loop
    //alignement des frame:reprojection (je reprojette la couleur)
    rs2::align alignV(RS2_STREAM_DEPTH);


    //point cloud
    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;



    while (1) // Application still alive?
    {
        // wait for frames and get frameset
        rs2::frameset frameset = pipeline.wait_for_frames();

        //lancement de l'alignement
        // alignement met du noir sur rgb et ir, ce qui bousille tout!!
        bool alignement = false;
        if (alignement) {
            frameset = alignV.process(frameset);
        }


        // pointeurs vers les frames
        rs2::video_frame ir_frame_left = frameset.get_infrared_frame(1);
        rs2::video_frame ir_frame_right = frameset.get_infrared_frame(2);
        rs2::video_frame color = frameset.get_color_frame();
        rs2::video_frame depth = frameset.get_depth_frame();

        //transfer vers openCV
        cv::Mat dMat_left = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)ir_frame_left.get_data());
        cv::Mat dMat_right = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)ir_frame_right.get_data());
        cv::Mat dMat_color = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)color.get_data());
        cv::Mat dMat_depth = cv::Mat(cv::Size(width, height), CV_16UC1, (void*)depth.get_data());

       
        
        cv::imshow("img_c", dMat_color);
        
        cv::imshow("img_d", dMat_depth*10);

       

        char c = cv::waitKey(1);
        if (c == 's')
        {
            if (cmpt==0){//creation du repertoire si on presse s: creation une seule fois
                
                bool ok=system(("mkdir -p data_"+timing).c_str());
                
            }
            //affichage intrnsics et extrinsics pour les 3 cameras
            intrinsics_extrinsics( pipeline_profile,RS2_STREAM_INFRARED ,1);
            intrinsics_extrinsics( pipeline_profile,RS2_STREAM_INFRARED ,2);
            intrinsics_extrinsics( pipeline_profile,RS2_STREAM_COLOR, 0);
            

            char ir1[10]="ir1"; char ir2[10]="ir2"; char rgb[10]="rgb";  char dep[10]="dep";
            
            string id= to_string(cmpt);
            writeImage(dMat_left,ir1,dir_time,id);
            writeImage(dMat_right,ir2,dir_time,id);
            

            writeImage(dMat_color,rgb,dir_time,id);

            writeImage(dMat_depth,dep,dir_time,id);


            // Generate the pointcloud and texture mappings
            pc.map_to(color);//map la texture sur le point cloud
            points = pc.calculate(depth);

            char namef[50];
            sprintf(namef,"%s/pc%s.ply",dir_time.c_str(),id.c_str());
            points.export_to_ply(namef, color);
            // transformation format pcl
            /*pcl_ptr cloud = points_to_pcl(points);

            char namef[50];
            sprintf(namef,"%s/pc%s.ply",dir_time.c_str(),id.c_str());
            pcl::io::savePLYFile(namef, *cloud); //enregistrer le pointcloud format ply on pourra passer au PCD ensuite
            */
            cmpt++;
        }
        else if (c == 'q')
            break;
    }


    return EXIT_SUCCESS;
}

