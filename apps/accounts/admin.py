from django.contrib import admin
from apps.accounts.models import UserProfile, UserActivityLog


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "organization", "avatar_color", "created_at")
    list_filter = ("role", "created_at")
    search_fields = ("user__username", "user__email", "organization")


@admin.register(UserActivityLog)
class UserActivityLogAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "user", "action", "ip_address", "details")
    list_filter = ("action", "timestamp")
    search_fields = ("user__username", "details", "ip_address")
    readonly_fields = ("timestamp", "user", "action", "ip_address", "details")

