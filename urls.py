from django.urls import path
from .  import views

urlpatterns = [
    path('', views.home, name = 'home'),
    path('About/', views.About, name = 'About'),
    path('Partners/', views.Partners, name = 'Partners'),
    path('begin', views.begin, name = 'begin'),
    path('arm', views.arm, name = 'arm'),
    path('clust', views.clust, name = 'clust'),
    path('dropdown', views.dropdown, name = 'dropdown'),
    path('clust_result', views.clust_result, name = 'clust_result')
]