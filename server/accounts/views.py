from django.urls import path
from dj_rest_auth.registration.views import RegisterView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.response import Response
from accounts.serializers import SinglePasswordRegisterSerializer, CustomJWTSerializer

class CustomRegisterView(RegisterView):
    serializer_class = SinglePasswordRegisterSerializer

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        user = self.serializer_class.Meta.model.objects.get(username=request.data.get('username'))

        # JWT 토큰 생성
        refresh = RefreshToken.for_user(user)
        response.data['refresh'] = str(refresh)
        response.data['access'] = str(refresh.access_token)

        return response
    
class CustomLoginView(TokenObtainPairView):
    serializer_class = CustomJWTSerializer
