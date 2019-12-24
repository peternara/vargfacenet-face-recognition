| **Title**      |Face Recognition |
| ---------- |-------------------|
| **Team**       |Nguyễn Tuấn Việt - viet.nguyen@siliconprime.com |
| **Predicting** |We will build a model to Vietnamese celebrities face recognition. <br/>
	From a given data set of 1,000 people, teams need to build a model that predicts <br/> a new picture that corresponds to someone or from a completely new person.|
| **Data**       |link of data : https://drive.google.com/file/d/1kpxjaz3pIMrAhEjm7hJxcBsxKNhfl8t2/view |
| **Features**   | <ol> <li>image: continuous</li> <li>label: discrete</li> </ol>|
| **Models**     |<ol> <li>ARCface is a face cognition engine (Additive Angular Margin Loss). <br/> use ARCface as model face embedding. </li>  <li>Linear SVC (Support Vector Classifier) is to fit to the data you provide, <br/> returning a "best fit" hyperplane that divides, or categorizes, your data.<br/> From there, after getting the hyperplane, you can then feed some features <br/> to your classifier to see what the "predicted" class is. <br/> Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, <br/> so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.</li> </ol>|
| **Discussion** |Which model return the better resuls? <br/> How to improve the other models? <br/> How to tuning the hyper parameters in models? |
| **Future**     |Make image preprocessing and verify data.<br/> Use more libs as facenet, vgg_face2,...<br/> Make face recognition demo |
|**References**  |[1] https://www.aivivn.com/contests/2 <br/> [2]https://forum.machinelearningcoban.com/t/aivivn-face-recognition-1st-solution/4725/33 <br/> [3] https://forum.machinelearningcoban.com/t/aivivn-face-recognition-4th-solution/4734/3 <br/> [4] https://www.kaggle.com/arunkumarramanan/data-science-python-fuel-efficiency-prediction <br/> [5] https://github.com/deepinsight/insightface) |
| **Results**    |Measure method: MAP@5 <br/> Score: > 0.94 |