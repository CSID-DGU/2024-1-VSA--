from rest_framework.response import Response
from rest_framework.decorators import api_view

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, Http404
from django.conf import settings

from django.views.decorators.csrf import csrf_exempt
import os
import json

@csrf_exempt
def login_view(request):
    if request.method == "POST":
        try:
            # 요청 본문에서 JSON 데이터 읽기
            data = json.loads(request.body)
            user_id = data.get("username")
            password = data.get("password")

            # 유효성 검사
            if not user_id or not password:
                return JsonResponse({"error": "Missing fields"}, status=400)

            # 처리 로직 (예: 사용자 생성, 데이터베이스 저장 등)
            # 여기에 실제 비즈니스 로직을 추가하세요.
            return JsonResponse({"success": True, "message": "User registered successfully!"})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    else:
        return JsonResponse({"error": "Invalid method"}, status=405)

@api_view(['GET'])
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

@api_view(['GET'])
def getVideoList(request):
    if request.method == 'GET':

        result = {
            "code": "200",
            "result": "SUCCESS",
            "message": "종목 코드 조회에 성공했습니다.",
            "data": [
                {
                    "id": 1,
                    "video_file": "https://a2b1.s3.us-east-1.amazonaws.com/videos/KakaoTalk_20240229_151332945.mp4",
                    "processed_video_file": "null",
                    "status": "Pending",
                    "created_at": "2024-12-25T18:45:44.610474+09:00"
                }
            ]
        }

        return Response(result)
    
@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('file')
        if video_file:
            save_path = os.path.join('uploads', video_file.name)
            with open(save_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
            return JsonResponse({'message': '업로드 성공!', 'filename': video_file.name}, status=200)
        return JsonResponse({'error': '파일 업로드 실패'}, status=400)
    return JsonResponse({'error': 'GET 요청은 지원되지 않습니다.'}, status=405)

def download_video(request, filename):
    file_path = os.path.join('uploads/', filename)  # 파일 경로 설정

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type='video/mp4')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
    else:
        raise Http404("파일을 찾을 수 없습니다.")