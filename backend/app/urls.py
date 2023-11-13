from garpixcms.urls import *  # noqa
from django.urls import path


urlpatterns = [
                  path('api/', include('modelthreed.urls'))
              ] + urlpatterns  # noqa
