"""
Security decorators for ML Studio views.
"""

from functools import wraps
from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.shortcuts import redirect


def studio_login_required(view_func):
    """
    Ensures that the user is authenticated.
    If not, redirects to /auth/login/?next=... with a warning flash message.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(
                request,
                "Vous devez être connecté pour accéder à l'espace de travail ML Studio.",
            )
            return redirect(f"/auth/login/?next={request.path}")
        return view_func(request, *args, **kwargs)

    return _wrapped_view


def admin_required(view_func):
    """
    Ensures that the user is authenticated and is an administrator (is_staff or is_superuser).
    If not authenticated, redirects to login.
    If authenticated but not admin, raises 403 PermissionDenied.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(f"/auth/login/?next={request.path}")
        if not (request.user.is_staff or request.user.is_superuser):
            raise PermissionDenied("Accès réservé aux administrateurs de ML Studio.")
        return view_func(request, *args, **kwargs)

    return _wrapped_view


def anonymous_only(view_func):
    """
    Prevents already logged-in users from visiting login or register pages.
    Redirects them to dashboard.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect("core:dashboard")
        return view_func(request, *args, **kwargs)

    return _wrapped_view

