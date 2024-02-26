from django.contrib import admin

from mailbox_web.models import User, Email


admin.site.register(User)
admin.site.register(Email)
