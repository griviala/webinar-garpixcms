from rest_framework import serializers


class ParseImageSerializer(serializers.Serializer):

    images = serializers.ListField(child=serializers.ImageField(), required=True)
    csv_file = serializers.FileField(required=True)
