# ARCface + SVM cho bài toán nhận diện người nổi tiếng
Mã nguồn chương trình đạt hạng 4 cuộc thi nhận diện người nổi tiếng do AIVIVN tổ chức

## Requirements
- python==3.5.2
- scikit-learn==0.20.3
- pandas==0.23.1
- scikit-image==0.14.2
- scipy==1.0.0
- mxnet==1.4.0.post0

## Giải pháp
### Xử lý dữ liệu
dữ liệu gồm 1000 nhãn trong đó số lượng ảnh của từng nhãn trong khoảng từ 1-16, trong đó có một vài nhãn còn thuộc cùng 1 người, và có ảnh bị gán nhãn sai trong dữ liệu. Tuy nhiên mình không xử lý gì với 2 vấn đề này mà chỉ thực hiện augment để thêm dữ liệu cho những nhãn có < 3 ảnh, sử dụng xoay, thêm nhiễu và flip ảnh.

### Face Embedding
Face embedding mình sử dụng ở đây là arcface (theo repo: https://github.com/deepinsight/insightface)

### Mô hình
- Mình coi bài toán như một bài phân loại, và sử dụng ngưỡng trên đầu ra để dự đoán người lạ (nhãn 1000)
- Mô hình mình lựa chọn là SVM (sklearn.linearSVC), có tunning để lựa chọn tham số phù hợp với dữ liệu
- Ngưỡng để xác định người lạ được lựa chọn trên giá trị score của decision_function trong sklearn.linearSVC. Mình thực hiện submit một vài lần để dự đoán ra số người lạ, rồi sau đó xác định ngưỡng dựa theo số lượng người lạ mà mình dự đoán (cuối cùng chọn threshold = -0.55). Với những ảnh có score <= ngưỡng, nhãn 1000 được đẩy lên đầu và tiếp đến là 4 nhãn có score cao nhất, với trường hợp score > ngưỡng, nhãn 1000 đưọc đặt tại vị trí 3.

### Sử dụng thêm dữ liệu từ tập test
Mình sử dụng mô hình tốt nhất đã có để dự đoán nhãn trên tập public test, rồi dùng một phần dữ liệu mà mô hình dự đoán với độ chính xác cao để làm dữ liệu thêm (~14k ảnh), việc này gây thêm nhiễu cho mô hình nhưng lại tăng được đáng kể dữ liệu. Sau đó mình train lại mô hình với tập train + dữ liệu thêm này, kết quả thu được tốt hơn so với trước đó.

## Chạy chương trình
Clone insightface
```
git clone https://github.com/deepinsight/insightface.git
```

Sinh embedding
```
python3 prepare_data.py --data_path="thư mục data cuộc thi"
```

```
python3 augment_data.py
```

Note: find . -name "*.DS_Store" -type f -delete

```
python3 gen_emb.py
```

Huấn luyện mô hình
```
python3 arcface+linearSVC.py --mode="normal"
```

Thêm dữ liệu
```
python3 add_data.py
```

Huấn luyện lại mô hình với dữ liệu thêm
```
python3 arcface+linearSVC.py --mode="add"


##### Summary Table

<!---
| Item     | Meaning|
| ---------- |-------------------|
| **Author**| 
| **Title**      | Give the project a title if not given|
| **Topics**       | In which field? CV, NLP, Transfer learning, etc. Using which algorithms? DCNN, LSTM, etc.|
| **Descriptions**       | What the project is about?|
| **Links**       | Github repo, medium/blog, etc.|
| **Framework**       | Scikit-learn, TF, Keras, PyTorch, NLTK, etc.|
| **Pretrained Models**       | Available or not, size.|
| **Datasets**       | Dataset used, available or not, size, links.|
| **Level of difficulty**       | Easy/small; intermediate/take half a day to run; advanced/big, days to run.|


## Example
--->
|      | |
| ---------- |-------------------|
| **Author**       | Nguyễn Tuấn Việt
					viet.nguyen@siliconprime.com|
| **Title**        | Face recognition |
| **Topics**       | Ứng dụng trong computer vision, sử dụng thuật toán chính là image embedding and linearSVC|
| **Descriptions** | Input sẽ là các tấm hình và file .txt có tên tương ứng và chứa 5 thông số của object. đầu tiên là ```<class object>``` và ```<x, y, width, height>``` của bounding box chứa vật. khi train xong sẽ trả ra output là file trọng số ```weights```. Ta sẽ sử dụng trọng số ```weights``` đã train để predict bounding box và class của các object trong hình|
| **Links**        | https://github.com/ultralytics/yolov3|
| **Framework**    | PyTorch|
| **Pretrained Models**  | sử dụng weight đã được train sẵn https://pjreddie.com/media/files/yolov3.weights|
| **Datasets**     |Mô hình được train với bộ dữ liệu cocodataset.org. Ngoài ra còn có các tập dữ liệu có thể sử dụng: PASCAL VOC, Open Images Dataset V4,..v.v.,|
| **Level of difficulty**|Sử dụng nhanh và dễ, có thể train lại với tập dữ liệu khác tốc độ tùy thuộc vào phần cứng và hình ảnh input|



```