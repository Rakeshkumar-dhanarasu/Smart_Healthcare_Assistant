from paddleocr import PaddleOCR

detect_model = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False,
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="PP-OCRv5_server_rec"
    )

image_path = "C:/Users/Rakesh Kumar/VSCode/Medical_Assistant/datasets/test_images/tester1.jpg" 

result = detect_model.predict(image_path)

# Print bounding boxes
for res in result:  
    print(res['rec_texts'])