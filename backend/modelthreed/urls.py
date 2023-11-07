from rest_framework import routers

from modelthreed import views

router = routers.DefaultRouter()
router.register('', views.ImageParserView, basename='parse_images')

urlpatterns = router.urls
