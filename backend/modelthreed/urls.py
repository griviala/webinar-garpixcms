from modelthreed import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register('', views.ImageParserView, basename='parse_image')

urlpatterns = router.urls
