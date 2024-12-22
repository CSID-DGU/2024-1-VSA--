from dj_rest_auth.registration.serializers import RegisterSerializer
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()

class SinglePasswordRegisterSerializer(RegisterSerializer):
    # 비밀번호 확인 제거
    password2 = None

    class Meta:
        model = User
        fields = ('id', 'username', 'password1')

    def validate(self, data):
        if 'password1' not in data:
            raise serializers.ValidationError({"password1": "Password is required."})
        return data

    def save(self, request):
        user = super().save(request)
        user.set_password(self.validated_data['password1'])
        user.save()
        return user


class CustomJWTSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)
        return {
            "id": self.user.id,
            "username": self.user.username,
            "access": data["access"],
            "refresh": data["refresh"],
        }


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'profile_image']
        read_only_fields = ['id', 'username']