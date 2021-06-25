python demo.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
  --video-input ../../rekall/data/video-clip.mp4 \
  --output video-output.mp4 --confidence-threshold 0.6 \
  --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl