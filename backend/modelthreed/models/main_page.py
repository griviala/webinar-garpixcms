from django.db import models
from garpix_page.models import BasePage


class MainPage(BasePage):
    template = "pages/main.html"

    class Meta:
        verbose_name = "Главная страница"
        verbose_name_plural = "Главные страницы"
        ordering = ("-created_at",)
