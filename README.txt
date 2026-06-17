MediAI Chest - ResNet101 Chest X-ray Diagnosis System
1. Giới thiệu
    MediAI Chest là hệ thống web hỗ trợ chẩn đoán bệnh phổi từ ảnh X-quang ngực bằng mô hình học sâu. Hệ thống sử dụng mô hình ResNet101 để phân loại ảnh X-quang phổi thành bốn lớp: COVID-19, Normal, Pneumonia và Tuberculosis.
    Ngoài việc hiển thị kết quả dự đoán, hệ thống còn tích hợp các kỹ thuật Explainable AI (XAI) nhằm hỗ trợ giải thích vùng ảnh có ảnh hưởng đến quyết định của mô hình. Các kỹ thuật XAI được sử dụng gồm Grad-CAM, Grad-CAM++, Integrated Gradients và Occlusion Sensitivity.
    Hệ thống được xây dựng phục vụ mục đích nghiên cứu, học tập và minh họa quá trình ứng dụng AI trong phân tích ảnh y khoa. Kết quả dự đoán chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.
2. Chức năng chính
    Tải ảnh X-quang phổi từ người dùng.
    Hiển thị ảnh đầu vào trước khi phân tích.
    Dự đoán ảnh thuộc một trong bốn lớp:
    COVID-19, Normal, Pneumonia, Tuberculosis
    Hiển thị độ tin cậy của kết quả dự đoán.
    Hiển thị xác suất của từng lớp bệnh.
    Minh họa kết quả giải thích bằng các kỹ thuật XAI.
    Lưu lịch sử các lần phân tích ảnh.
    Trình bày quy trình xử lý của hệ thống AI.
3. Mô hình sử dụng
    Mô hình chính của hệ thống là ResNet101. Đây là mô hình mạng nơ-ron tích chập sâu có sử dụng cơ chế residual learning, giúp cải thiện khả năng học đặc trưng trong các mạng có nhiều lớp.
    Trong đề tài này, ResNet101 được sử dụng để trích xuất đặc trưng từ ảnh X-quang phổi và phân loại ảnh thành bốn nhóm bệnh. Kết quả đánh giá trên tập kiểm thử đạt độ chính xác khoảng 95%.
    4. Kỹ thuật Explainable AI
    Hệ thống tích hợp các kỹ thuật Explainable AI nhằm làm rõ hơn quá trình ra quyết định của mô hình:
    Grad-CAM: tạo bản đồ nhiệt thể hiện vùng ảnh quan trọng đối với kết quả dự đoán.
    Grad-CAM++: cải thiện khả năng định vị vùng ảnh có ảnh hưởng đến dự đoán.
    Integrated Gradients: phân tích mức độ đóng góp của từng vùng ảnh dựa trên gradient tích lũy.
    Occlusion Sensitivity: che khuất từng vùng ảnh để đánh giá mức độ ảnh hưởng đến xác suất dự đoán.
    Các bản đồ XAI chỉ cho biết vùng ảnh có ảnh hưởng đến mô hình, không được xem là kết luận y khoa.