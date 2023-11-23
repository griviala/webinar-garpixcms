from django.db import models
from garpix_page.models import BasePage


class MainPage(BasePage):
    template = "pages/main.html"
    login_required = True

    class Meta:
        verbose_name = "Главная"
        verbose_name_plural = "Главные"
        ordering = ("-created_at",)
