"""
URL routing for accounts app.
"""

from django.contrib.auth import views as auth_views
from django.urls import path, reverse_lazy
from apps.accounts import views

app_name = "accounts"

urlpatterns = [
    # Authentication routes
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("profile/", views.profile_view, name="profile"),
    path("profile/delete/", views.delete_account_view, name="delete_account"),

    # HTMX live validations
    path("check-username/", views.check_username_view, name="check_username"),
    path("check-email/", views.check_email_view, name="check_email"),

    # Password Reset (Django standard views styled with Omakase templates)
    path(
        "password-reset/",
        auth_views.PasswordResetView.as_view(
            template_name="accounts/password_reset.html",
            email_template_name="accounts/password_reset_email.html",
            subject_template_name="accounts/password_reset_subject.txt",
            success_url=reverse_lazy("accounts:password_reset_done"),
        ),
        name="password_reset",
    ),
    path(
        "password-reset/done/",
        auth_views.PasswordResetDoneView.as_view(
            template_name="accounts/password_reset_done.html"
        ),
        name="password_reset_done",
    ),
    path(
        "password-reset-confirm/<uidb64>/<token>/",
        auth_views.PasswordResetConfirmView.as_view(
            template_name="accounts/password_reset_confirm.html",
            success_url=reverse_lazy("accounts:password_reset_complete"),
        ),
        name="password_reset_confirm",
    ),
    path(
        "password-reset-complete/",
        auth_views.PasswordResetCompleteView.as_view(
            template_name="accounts/password_reset_complete.html"
        ),
        name="password_reset_complete",
    ),

    # Admin SaaS User Management
    path("administration/users/", views.admin_users_view, name="admin_users"),
    path("administration/users/table/", views.admin_users_table_partial, name="admin_users_table"),
    path("administration/users/<int:user_id>/toggle-role/", views.admin_toggle_role_view, name="admin_toggle_role"),
    path("administration/users/<int:user_id>/toggle-status/", views.admin_toggle_status_view, name="admin_toggle_status"),
    path("administration/users/create/", views.admin_create_user_view, name="admin_create_user"),
    path("administration/users/<int:user_id>/delete/", views.admin_delete_user_view, name="admin_delete_user"),
]

