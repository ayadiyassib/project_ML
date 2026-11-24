"""
Access control middleware for ML Studio.
Ensures private studio routes are only accessible to authenticated users.
"""

from __future__ import annotations

from django.shortcuts import redirect
from django.urls import resolve

# Routes accessible to anonymous guests
PUBLIC_NAMESPACES = {"accounts"}
PUBLIC_URL_NAMES = {
    "landing",
    "login",
    "register",
    "password_reset",
    "password_reset_done",
    "password_reset_confirm",
    "password_reset_complete",
    "check_username",
    "check_email",
}


class StudioAccessControlMiddleware:
    """
    Middleware that enforces authentication across the ML Studio workspace.
    Public routes (landing, login, register, password reset, static, media, admin)
    are permitted for guests; all other routes redirect to login.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path_info

        # Always permit assets, Django admin, and root landing page
        if (
            path.startswith("/admin/")
            or path.startswith("/static/")
            or path.startswith("/media/")
            or path == "/"
        ):
            return self.get_response(request)

        try:
            resolver_match = resolve(path)
            url_name = resolver_match.url_name
            namespace = resolver_match.namespace
        except Exception:
            url_name = ""
            namespace = ""

        # Check if the matched route is designated as public
        is_public = (
            url_name in PUBLIC_URL_NAMES
            or (namespace in PUBLIC_NAMESPACES and not path.startswith("/auth/administration/"))
        )

        # Administration pages require admin staff rights
        if path.startswith("/administration/") or path.startswith("/auth/administration/"):
            if not request.user.is_authenticated:
                return redirect(f"/auth/login/?next={path}")

        # If private route and user is not authenticated, redirect to login
        if not is_public and not request.user.is_authenticated:
            return redirect(f"/auth/login/?next={path}")

        return self.get_response(request)

