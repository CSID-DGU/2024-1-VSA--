from django.urls import path, include
from .views import CustomRegisterView, CustomLoginView

urlpatterns = [
    path('register/', CustomRegisterView.as_view(), name='custom_rest_register'),
    path('login/', CustomLoginView.as_view(), name='custom_login'),

]