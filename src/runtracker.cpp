#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ctime>

#include "kcftracker.hpp"

#include <dirent.h>

#include <boost/program_options.hpp>

#include "json.hpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;
using json = nlohmann::json;

string frame_path(const json video, const int frame_id) {
    for (int i = 0;  i < video["frames"].size(); i++) {
        json frame = video["frames"][i];
        if (frame["frame"].get<int>() == frame_id) {
            return video["root_path"].get<string>() + "/" + frame["path"].get<string>();
        }
    }
    return "";
}

int single_tracking(KCFTracker & tracker,
    const json & video, const int start_frame, const int length,
    const float xMin, const float yMin, const float x2, const float y2,
    json & tracklet) {

    // Frame readed
    Mat frame;

    // Tracker results
    Rect result;

    // Using min and max of X and Y for groundtruth rectangle
    float width = x2 - xMin;
    float height = y2 - yMin;

    // Frame counter
    int nFrames = 0;


    for (int i = 0; i < length; i++) {
        int frame_id = start_frame + i;
        string frameName = frame_path(video, frame_id);

        if (frameName == "") return i;

        // Read each frame from the list
        frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

        // current box
        json curframe;
        curframe["frame"] = frame_id;
        curframe["anchor"] = i;
        curframe["scores"] = 1.0;

        // First frame, give the groundtruth to the tracker
        if (nFrames == 0) {
            tracker.init( Rect(xMin, yMin, width, height), frame );
            rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
            curframe["bbox"] = {xMin, yMin, xMin + width, yMin + height};
        }
        // Update
        else{
            result = tracker.update(frame);
            rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
            curframe["bbox"] = {result.x, result.y, result.x + result.width, result.y + result.height};
        }

        nFrames++;

        if (0){
            imshow("Image", frame);
            waitKey(1);
        }
        tracklet.push_back(curframe);
    }
    return length;
}

int main(int argc, char* argv[]){

    po::options_description desc("KCF tracker for VID.");
    desc.add_options()
        ("help", "produce help message")
        ("video", po::value<string>(), "input video")
        ("proposals", po::value<string>(), "proposal file")
        ("output", po::value<string>(), "output file")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    // read inputs
    std::ifstream video_file(vm["video"].as<string>());
    std::string video_contents( (std::istreambuf_iterator<char>(video_file) ),
                                (std::istreambuf_iterator<char>()    ) );
    json video = json::parse(video_contents);

    std::ifstream box_file(vm["proposals"].as<string>());
    std::string box_contents(  (std::istreambuf_iterator<char>(box_file) ),
                               (std::istreambuf_iterator<char>()    ) );
    json boxes = json::parse(box_contents);

    std::ofstream output_file(vm["output"].as<string>());


    // setting some parameters
    const int length = 20;
    const int sample_rate = length;

    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool SILENT = true;
    bool LAB = false;

    // Create KCFTracker object
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    json track_proto;
    track_proto["video"] = video["video"];
    track_proto["method"] = "KCF";
    json tracks;
    int count = 0;
    long long int frame_count = 0;
    const clock_t begin_time = clock();
    for (int i = 0; i < boxes["boxes"].size(); i++) {
        json curbox = boxes["boxes"][i];
        int frame_id = curbox["frame"].get<int>();
        if (frame_id % sample_rate != 1) continue;
        count++;
        json bbox = curbox["bbox"];
        json tracklet;
        int num_frame = single_tracking(tracker, video,
            frame_id, length,
            bbox[0].get<float>(), bbox[1].get<float>(), bbox[2].get<float>(), bbox[3].get<float>(),
            tracklet);
        frame_count += num_frame;
        tracks.push_back(tracklet);
    }
    track_proto["tracks"] = tracks;
    cout << "Write " << count << " tracks to output file: " << vm["output"].as<string>() << endl;
    float total_time = float(clock() - begin_time) / CLOCKS_PER_SEC;
    cout << "Total time: " << total_time << " s. " << frame_count / total_time << " fps." << endl;
    output_file << track_proto.dump();
}
