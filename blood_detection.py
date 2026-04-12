import cv2
from ultralytics import YOLO

# 1. 載入你訓練好的模型 (請將 'blood_model.pt' 換成你的檔案名稱)
model = YOLO('best.pt')

# 2. 開啟攝影機 (0 代表預設的筆電鏡頭，如果是影片請填入 '影片路徑.mp4')
cap = cv2.VideoCapture('blood.mp4')

while cap.isOpened():
    ret, frame = cap.read() # 讀取當下畫面
    if not ret:
        break

    # 3. 讓模型進行預測
    results = model(frame)

    # 4. 解析結果並打馬賽克
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 取得血跡的座標 (左上角 x1, y1, 右下角 x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 擷取畫面中的「血跡區域」
            blood_region = frame[y1:y2, x1:x2]

            # === 馬賽克處理手法 ===
            # 方法 A：高斯模糊 (較柔和的霧化感)
            blurred_region = cv2.GaussianBlur(blood_region, (51, 51), 0)
            
            # 方法 B：傳統格子馬賽克 (將圖片縮得很小再放大，產生像素顆粒感)
            # h, w = blood_region.shape[:2]
            # small = cv2.resize(blood_region, (w//10, h//10)) # 縮小 10 倍
            # blurred_region = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

            # 將處理後的區域「貼回」原本的畫面上
            frame[y1:y2, x1:x2] = blurred_region

    # 顯示最終處理好的畫面
    cv2.imshow('Real-time Blood Mosaic', frame)

    # 按下鍵盤的 'q' 鍵可以退出程式
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機資源並關閉視窗
cap.release()
cv2.destroyAllWindows()