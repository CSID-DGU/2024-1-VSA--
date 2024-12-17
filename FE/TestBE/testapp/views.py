from rest_framework.response import Response
from rest_framework.decorators import api_view

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, Http404
from django.conf import settings

from django.views.decorators.csrf import csrf_exempt
import os

@api_view(['GET'])
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

@api_view(['GET'])
def getVideoList(request):
    if request.method == 'GET':
        item = os.listdir("uploads")
        result = {
            "code": "200",
            "result": "SUCCESS",
            "message": "종목 코드 조회에 성공했습니다.",
            "data": {
                "videos": item
            }
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