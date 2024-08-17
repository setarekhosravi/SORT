# YOLOv5 Integration with SORT Tracker for Real-time Object Tracking

<p align="justify">
This project demonstrates the integration of the SORT (Simple Online and Realtime Tracking) algorithm with YOLOv5 for efficient and accurate object tracking. By combining YOLOv5's robust object detection capabilities with SORT's lightweight tracking, this implementation provides a seamless solution for real-time multi-object tracking. The system is designed to detect and track objects across frames in a video stream, maintaining high performance while being easy to use and modify for various applications.
</p>

# Run on Image Sequences

1. To run the object tracking with YOLOv5 and SORT, follow these steps:

```bash
git clone https://github.com/setarekhosravi/SORT.git
cd SORT
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Then clone and install yolov5 repository:

```bash
pip install ultralytics
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

4. Run the tracking example:
```bash
python track.py --weights /path/to/yolov5/weights --input_type image --input_path /path/to/image/sequence --save_mot true --save_path /save/path --save_video false
                                                               video (not implemented)                               false                                    true
```
