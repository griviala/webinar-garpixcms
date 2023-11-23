from django.db import models
from garpix_page.models import BasePage


class LoginPage(BasePage):
    template = "pages/login.html"

    class Meta:
        verbose_name = "Авторизация"
        verbose_name_plural = "Авторизация"
        ordering = ("-created_at",)
