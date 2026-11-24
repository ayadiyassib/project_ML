"""
Service layer for authentication, user management, and activity logging.
Keeps views thin and pure controllers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.db.models import Count, Q, Sum
from django.http import HttpRequest

from apps.accounts.models import UserActivityLog, UserProfile
from apps.datasets.models import Dataset, human_size
from apps.ml_engine.models import TrainedModel


def get_client_ip(request: HttpRequest) -> Optional[str]:
    """Extract client IP address from HTTP request headers."""
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def log_user_activity(
    user: User,
    action: str,
    ip: Optional[str] = None,
    details: str = "",
) -> UserActivityLog:
    """Record an audit trail action for a user."""
    return UserActivityLog.objects.create(
        user=user,
        action=action,
        ip_address=ip,
        details=details,
    )


def authenticate_by_email_or_username(
    request: HttpRequest,
    identifier: str,
    password: str,
) -> Optional[User]:
    """
    Authenticate a user using either their username or email address.
    """
    identifier = (identifier or "").strip()
    if not identifier or not password:
        return None

    # Find matching user by username or email
    user_qs = User.objects.filter(
        Q(username__iexact=identifier) | Q(email__iexact=identifier)
    )
    user_obj = user_qs.first()
    if not user_obj:
        return None

    # Authenticate against Django's backend with the canonical username
    return authenticate(request, username=user_obj.username, password=password)


def register_new_user(
    data: Dict[str, Any],
    request: Optional[HttpRequest] = None,
) -> User:
    """
    Create a new User and assign UserProfile defaults.
    """
    user = User.objects.create_user(
        username=data["username"].strip(),
        email=data["email"].strip().lower(),
        password=data["password"],
        first_name=data.get("first_name", "").strip(),
        last_name=data.get("last_name", "").strip(),
    )
    ip = get_client_ip(request) if request else None
    log_user_activity(user, UserActivityLog.ActionChoices.LOGIN, ip=ip, details="Inscription réussie")
    return user


def get_user_stats(user: User) -> Dict[str, Any]:
    """
    Aggregate personal metrics for a user: datasets, trained models, disk usage.
    """
    user_datasets = Dataset.objects.filter(user=user)
    datasets_count = user_datasets.count()
    models_count = TrainedModel.objects.filter(user=user).count()
    total_bytes = user_datasets.aggregate(total=Sum("size_bytes"))["total"] or 0

    return {
        "datasets_count": datasets_count,
        "models_count": models_count,
        "storage_bytes": total_bytes,
        "storage_human": human_size(total_bytes),
    }


def admin_filter_users(
    query: str = "",
    role_filter: str = "",
    status_filter: str = "",
):
    """
    Query and filter users for the admin management view.
    Annotates dataset and model counts.
    """
    users = User.objects.select_related("profile").prefetch_related(
        "datasets", "trained_models"
    ).order_by("-date_joined")

    if query:
        query = query.strip()
        users = users.filter(
            Q(username__icontains=query)
            | Q(email__icontains=query)
            | Q(first_name__icontains=query)
            | Q(last_name__icontains=query)
        )

    if role_filter == "admin":
        users = users.filter(Q(is_staff=True) | Q(is_superuser=True))
    elif role_filter == "user":
        users = users.filter(is_staff=False, is_superuser=False)

    if status_filter == "active":
        users = users.filter(is_active=True)
    elif status_filter == "inactive":
        users = users.filter(is_active=False)

    return users


def admin_toggle_user_role(target_user: User) -> str:
    """
    Toggle between Administrator and Standard Member.
    Returns the new role name.
    """
    profile = getattr(target_user, "profile", None)
    if target_user.is_staff or target_user.is_superuser:
        target_user.is_staff = False
        target_user.is_superuser = False
        target_user.save(update_fields=["is_staff", "is_superuser"])
        if profile:
            profile.role = UserProfile.RoleChoices.USER
            profile.save(update_fields=["role"])
        return "Membre"
    else:
        target_user.is_staff = True
        target_user.save(update_fields=["is_staff"])
        if profile:
            profile.role = UserProfile.RoleChoices.ADMIN
            profile.save(update_fields=["role"])
        return "Administrateur"


def admin_toggle_user_status(target_user: User) -> bool:
    """
    Toggle user active status.
    Returns the new is_active boolean.
    """
    target_user.is_active = not target_user.is_active
    target_user.save(update_fields=["is_active"])
    return target_user.is_active


def admin_create_user(data: Dict[str, Any]) -> User:
    """
    Create a new user from the Admin SaaS modal.
    """
    role = data.get("role", "USER")
    is_admin = role == "ADMIN"
    user = User.objects.create_user(
        username=data["username"].strip(),
        email=data["email"].strip().lower(),
        password=data["password"],
        first_name=data.get("first_name", "").strip(),
        last_name=data.get("last_name", "").strip(),
        is_staff=is_admin,
    )
    if hasattr(user, "profile"):
        user.profile.role = UserProfile.RoleChoices.ADMIN if is_admin else UserProfile.RoleChoices.USER
        if "organization" in data and data["organization"]:
            user.profile.organization = data["organization"].strip()
        user.profile.save()
    return user

