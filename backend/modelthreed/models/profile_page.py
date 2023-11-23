from django.db import models
from garpix_page.models import BasePage


class ProfilePage(BasePage):
    template = "pages/profile.html"

    login_required = True

    class Meta:
        verbose_name = "Профиль"
        verbose_name_plural = "Профиль"
        ordering = ("-created_at",)

    def get_context(self, request=None, *args, **kwargs):

        context = super().get_context(request, *args, **kwargs)

        user = request.user

        if user.is_authenticated:
            context.update({
                'current_user': {
                    'first_name': user.first_name,
                    'last_name': user.last_name
                }
            })

        return context
