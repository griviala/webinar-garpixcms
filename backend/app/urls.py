from garpixcms.urls import *  # noqa

urlpatterns = [
    path('api/', include('modelthreed.urls'))
] + urlpatterns
