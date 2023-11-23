from ..models.profile_page import ProfilePage
from django.contrib import admin
from garpix_page.admin import BasePageAdmin


@admin.register(ProfilePage)
class ProfilePageAdmin(BasePageAdmin):
    pass
