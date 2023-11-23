from ..models.main_page import MainPage
from django.contrib import admin
from garpix_page.admin import BasePageAdmin


@admin.register(MainPage)
class MainPageAdmin(BasePageAdmin):
    pass
