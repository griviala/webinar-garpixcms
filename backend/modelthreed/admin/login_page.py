from ..models.login_page import LoginPage
from django.contrib import admin
from garpix_page.admin import BasePageAdmin


@admin.register(LoginPage)
class LoginPageAdmin(BasePageAdmin):
    pass
