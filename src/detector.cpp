#include "detector.h"


Detector::Detector(std::string& modelName, int width, int height, float nms_threshold,
        float score_threshold):Model(modelName, width, height){
    mVariance.push_back(0.1);
    mVariance.push_back(0.1);
    mVariance.push_back(0.2);
    mVariance.push_back(0.2);
    mScoreThreshold = score_threshold;
    mNMSThreshold   = nms_threshold;


    PrepareInputAndOutputNames();
    SetUpInputAndOutputTensors();
}

void Detector::PrepareInputAndOutputNames(){
    mOutputNames.push_back("cls_logits");
    mOutputNames.push_back("bbox_preds");
    mOutputNames.push_back("anchors");
}

void Detector::Preprocess(const cv::Mat raw_image, cv::Mat& image){
    cv::cvtColor(raw_image,image, cv::COLOR_BGR2RGB);

    mOriginInputSize.push_back(raw_image.rows);
    mOriginInputSize.push_back(raw_image.cols);
    // order? hw or wh
    cv::resize(image, image, cv::Size(mInputSize[1], mInputSize[0]));

    image.convertTo(image, CV_32FC3);
    const float mean_vals[3] = { 123.f, 117.f, 104.f};
    image = image - cv::Scalar(mean_vals[0], mean_vals[1], mean_vals[2]);
}






void Detector::GetTopK(std::vector<BoxInfo>& input, std::vector<BoxInfo>& output, int top_k)
{
    std::sort(input.begin(), input.end(),
            [](const BoxInfo& a, const BoxInfo& b)
            {
            return a.score > b.score;
            });

    if (input.size() > top_k) {
        for (int k = 0; k < top_k; k++) {
            output.push_back(input[k]);
        }
    }
    else {
        output = input;
    }
}

void Detector::NMS(std::vector<BoxInfo>& tmp_faces, std::vector<BoxInfo>& faces, float nms_threshold){
    int N = tmp_faces.size();
    std::vector<int> labels(N, -1);
    for(int i = 0; i < N-1; ++i)
    {
        for (int j = i+1; j < N; ++j)
        {
            cv::Rect pre_box = tmp_faces[i].box;
            cv::Rect cur_box = tmp_faces[j].box;
            float iou_ = iou(pre_box, cur_box);
            if (iou_ > nms_threshold) {
                labels[j] = 0;
            }
        }
    }

    for (int i = 0; i < N; ++i)
    {
        if (labels[i] == -1)
            faces.push_back(tmp_faces[i]);
    }
}

void Detector::GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold){
    auto tensors_host = mOutputTensorsHost;

    auto scores_dataPtr  = tensors_host[0]->host<float>();
    auto boxes_dataPtr   = tensors_host[1]->host<float>();
    auto anchors_dataPtr = tensors_host[2]->host<float>();
    int num_boxes = tensors_host[0]->channel();
    int raw_image_width = mOriginInputSize[1];
    int raw_image_height = mOriginInputSize[0];

    for(int i = 0; i < num_boxes; ++i)
    {
        // location decoding
        float ycenter =     boxes_dataPtr[i*4 + 1] * mVariance[1]  * anchors_dataPtr[i*4 + 3] + anchors_dataPtr[i*4 + 1];
        float xcenter =     boxes_dataPtr[i*4 + 0] * mVariance[0]  * anchors_dataPtr[i*4 + 2] + anchors_dataPtr[i*4 + 0];
        float h       = exp(boxes_dataPtr[i*4 + 3] * mVariance[3]) * anchors_dataPtr[i*4 + 3];
        float w       = exp(boxes_dataPtr[i*4 + 2] * mVariance[2]) * anchors_dataPtr[i*4 + 2];

        float ymin    = ( ycenter - h * 0.5 ) * raw_image_height;
        float xmin    = ( xcenter - w * 0.5 ) * raw_image_width;
        float ymax    = ( ycenter + h * 0.5 ) * raw_image_height;
        float xmax    = ( xcenter + w * 0.5 ) * raw_image_width;

        // probability decoding, softmax
        float nonface_prob = exp(scores_dataPtr[i*2 + 0]);
        float face_prob    = exp(scores_dataPtr[i*2 + 1]);

        float ss           = nonface_prob + face_prob;
        nonface_prob       /= ss;
        face_prob          /= ss;

        if (face_prob > score_threshold) {
            BoxInfo tmp_face;
            tmp_face.box.x = xmin;
            tmp_face.box.y = ymin;
            tmp_face.box.width  = xmax - xmin;
            tmp_face.box.height = ymax - ymin;
            tmp_face.score = face_prob;
            boxInfos.push_back(tmp_face);
        }
    }
}


void Detector::Detect(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos){
    // preprocess
    cv::Mat image;
    Preprocess(raw_image, image);


    Run(image);

    // postprocess
    std::vector<BoxInfo> boxInfos, boxInfos_left;
    GenerateBoxInfo(boxInfos, mScoreThreshold);
    // top k
    GetTopK(boxInfos, boxInfos_left, mTopK);
    // nms
    NMS(boxInfos_left, finalBoxInfos, mNMSThreshold);

    // handle corner case
}
