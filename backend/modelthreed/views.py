import json
import shutil

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

from modelthreed.serializers import ParseImagesSerializer
from modelthreed.source_code.model3d import Model3d


class ImageParserView(GenericViewSet):

    serializer_class = ParseImagesSerializer

    @action(methods=['POST'], detail=False)
    def parse_images(self, request, *args, **kwargs):

        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        images = serializer.validated_data['images']
        csv_file = serializer.validated_data['csv_file']

        csv_file_name = default_storage.save(csv_file.name, csv_file)

        for image in images:
            default_storage.save('/'.join(['images', image.name]), image)

        csv_file_url = '/'.join([settings.MEDIA_ROOT, csv_file_name])
        images_url = '/'.join([settings.MEDIA_ROOT, 'images/'])

        model = Model3d(imagesPath=images_url, boxQueue=csv_file_url)

        response_json = model.run()

        shutil.rmtree(images_url, ignore_errors=True)
        # shutil.rmtree(csv_file_url)

        response = HttpResponse(json.dumps(response_json), content_type='text/plain; charset=UTF-8')
        response['Content-Disposition'] = ('attachment; filename=response.json')

        return response
