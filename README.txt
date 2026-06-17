HƯỚNG DẪN CHẠY WEB TRÊN VISUAL STUDIO CODE

1. Giải nén thư mục medical_ai_xray_web_vscode.
2. Mở Visual Studio Code.
3. Chọn File > Open Folder và mở thư mục này.
4. Cài extension Live Server nếu chưa có.
5. Nhấn chuột phải vào file index.html.
6. Chọn Open with Live Server.

Ghi chú:
- Đây là giao diện web frontend dạng dashboard, chưa kết nối model AI thật.
- Nút Phân tích ảnh hiện đang mô phỏng kết quả dự đoán.
- Khi muốn kết nối model thật, cần thêm backend Flask/FastAPI để gọi file model .h5 hoặc .keras.
- Mô hình đang hiển thị mặc định: DenseNet121.
