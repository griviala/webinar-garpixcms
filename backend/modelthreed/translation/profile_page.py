from modeltranslation.translator import TranslationOptions, register
from ..models import ProfilePage


@register(ProfilePage)
class ProfilePageTranslationOptions(TranslationOptions):
    pass
