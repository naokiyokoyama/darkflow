from darkflow.net.build import TFNet
import cv2, glob

options = {"model": "cfg/etablev2.cfg", "load": "bin/cust_ev2_30000.weights", "threshold": 0.6, "labels": "cfg/etable.names"}
# options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights", "threshold": 0.4, "labels": "cfg/coco.names"}

tfnet = TFNet(options)

for img_path in glob.glob("sample_img/*.JPG"):
	imgcv = cv2.imread(img_path)
	result = tfnet.return_predict(imgcv)
	for box in result:
		imgcv = tfnet.drawdict(imgcv,box)
	cv2.imwrite("res/"+img_path.split('/')[-1],imgcv)