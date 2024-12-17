from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.video_service import get_camera_stream

router = APIRouter()

@router.get("/video-stream/")
def video_feed():
    # StreamingResponse로 get_camera_stream 함수를 호출하여 영상 스트리밍
    return StreamingResponse(get_camera_stream(), media_type='multipart/x-mixed-replace; boundary=frame')
