import json

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet
from modelthreed.serializers import ParseImageSerializer
from modelthreed.source_code.model3d import Model3d


class ImageParserView(GenericViewSet):

    serializer_class = ParseImageSerializer

    @action(methods=['POST'], detail=False)
    def parse_images(self, request, *args, **kwargs):

        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        for image in serializer.validated_data['images']:
            default_storage.save('/'.join(['images', image.name]), image)

        csv_name = default_storage.save(serializer.validated_data['csv_file'].name, serializer.validated_data['csv_file'])

        images_path = '/'.join([settings.MEDIA_ROOT, 'images/'])
        bos_queue = '/'.join([settings.MEDIA_ROOT, csv_name])

        model = Model3d(imagesPath=images_path, boxQueue=bos_queue)
        response_json = model.run()

        response = HttpResponse(json.dumps(response_json), content_type='text/plain; charset=UTF-8')
        response['Content-Disposition'] = ('attachment; filename=response.json')

        return response
