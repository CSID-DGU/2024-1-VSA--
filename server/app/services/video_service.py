import cv2

def get_camera_stream():
    camObj = cv2.VideoCapture(0)  # 카메라에서 실시간 영상 가져오기
    
    if not camObj.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera could not be opened\r\n\r\n'
        return
    
    while True:
        retVal, frame = camObj.read()  # 카메라에서 프레임 읽기
        if not retVal:
            break

        # 영상에 바운딩 박스 추가 (좌측 상단 모서리 좌표 (50, 50), 우측 하단 모서리 (200, 200))
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)  # 초록색 테두리

        # 영상에 텍스트 추가 (메시지: "Camera OK")
        cv2.putText(frame, 'Camera OK', (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 프레임을 JPEG 포맷으로 변환
        retVal, jpgImg = cv2.imencode('.jpg', frame)

        # 바이너리로 변환하여 스트리밍에 사용
        frame_bytes = jpgImg.tobytes()

        # 멀티파트 형식으로 프레임 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n\r\n')