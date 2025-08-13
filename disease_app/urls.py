from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_disease, name='home'),
    path('analyze/', views.predict_disease, name='predict_disease'),
    path('api/fertilizer-shops-mapbox', views.fertilizer_shops_mapbox, name='fertilizer_shops_mapbox'),
]
