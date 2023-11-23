from modeltranslation.translator import TranslationOptions, register
from ..models import LoginPage


@register(LoginPage)
class LoginPageTranslationOptions(TranslationOptions):
    pass
