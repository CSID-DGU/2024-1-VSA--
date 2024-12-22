from django.urls import path
from accounts.views import CustomRegisterView, CustomLoginView, UserProfileView

urlpatterns = [
    path('register/', CustomRegisterView.as_view(), name='custom-register'),
    path('login/', CustomLoginView.as_view(), name='custom-login'),
    path('profile/', UserProfileView.as_view(), name='user-profile'),
]
