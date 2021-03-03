import os 
import pickle
import json
import urllib3, requests, os, posixpath


with open("/home/ubuntu/CSE544-project/data/bdd100k/bbox_files/cabc30fc-e7726578.pkl", 'rb') as f:
    maskrcnn_bboxes_intel = pickle.load(f)
# VIDEO_COLLECTION_BASEURL_INTEL = "https://storage.googleapis.com/esper/dan_olimar/rekall_tutorials/cydet" 
# maskrcnn_bbox_files_intel = [ 'maskrcnn_bboxes_0001.pkl' ]


# for bbox_file in maskrcnn_bbox_files_intel:
#     req = requests.get(posixpath.join(VIDEO_COLLECTION_BASEURL_INTEL, bbox_file), verify=False)
#     maskrcnn_bboxes_intel = pickle.loads(req.content)

with open("/home/ubuntu/CSE544-project/rekall/data/bbox-2", 'w') as f:
    f.write(json.dumps(maskrcnn_bboxes_intel))