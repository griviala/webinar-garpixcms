from modeltranslation.translator import TranslationOptions, register
from ..models import MainPage


@register(MainPage)
class MainPageTranslationOptions(TranslationOptions):
    pass
