"""
Views and controllers for authentication, profile management, and admin user administration.
"""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import login, logout, update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse, reverse_lazy
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from apps.accounts import services
from apps.accounts.decorators import admin_required, anonymous_only, studio_login_required
from apps.accounts.forms import AdminCreateUserForm, LoginForm, RegisterForm, UserProfileForm
from apps.accounts.models import UserActivityLog, UserProfile
from apps.datasets.models import Dataset
from apps.ml_engine.models import TrainedModel


# ==============================================================================
# AUTHENTICATION FLOWS (Login, Register, Logout)
# ==============================================================================

@anonymous_only
def login_view(request: HttpRequest) -> HttpResponse:
    """Handle user login with username or email, remember-me support, and safe redirect."""
    next_url = request.GET.get("next") or request.POST.get("next") or ""

    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username_or_email = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            remember_me = form.cleaned_data.get("remember_me", False)

            user = services.authenticate_by_email_or_username(
                request, username_or_email, password
            )

            if user is not None:
                if not user.is_active:
                    messages.error(request, "Votre compte a été désactivé. Veuillez contacter un administrateur.")
                    return render(request, "accounts/login.html", {"form": form, "next": next_url})

                login(request, user)
                if remember_me:
                    request.session.set_expiry(86400 * 30)  # 30 days
                else:
                    request.session.set_expiry(0)  # Browser close

                services.log_user_activity(
                    user,
                    UserActivityLog.ActionChoices.LOGIN,
                    ip=services.get_client_ip(request),
                    details="Connexion réussie",
                )
                messages.success(request, f"Bon retour parmi nous, {user.first_name or user.username} !")

                # Validate safe redirect target
                if next_url and url_has_allowed_host_and_scheme(
                    next_url, allowed_hosts={request.get_host()}, require_https=request.is_secure()
                ):
                    return redirect(next_url)
                return redirect("core:dashboard")
            else:
                messages.error(request, "Identifiants invalides. Veuillez vérifier votre nom d'utilisateur et votre mot de passe.")
    else:
        form = LoginForm()

    return render(request, "accounts/login.html", {"form": form, "next": next_url})


@anonymous_only
def register_view(request: HttpRequest) -> HttpResponse:
    """Handle new user sign-up with instant login upon success."""
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = services.register_new_user(form.cleaned_data, request=request)
            login(request, user)
            messages.success(
                request,
                f"Bienvenue sur ML Studio, {user.first_name or user.username} ! Votre espace est prêt.",
            )
            return redirect("core:dashboard")
    else:
        form = RegisterForm()

    return render(request, "accounts/register.html", {"form": form})


def logout_view(request: HttpRequest) -> HttpResponse:
    """
    Secure logout using POST only.
    Purges session and redirects to public landing page.
    """
    if request.method == "POST":
        if request.user.is_authenticated:
            services.log_user_activity(
                request.user,
                UserActivityLog.ActionChoices.LOGOUT,
                ip=services.get_client_ip(request),
                details="Déconnexion utilisateur",
            )
        logout(request)
        request.session.flush()
        messages.info(request, "Vous avez été déconnecté avec succès. À bientôt !")
        return redirect("core:landing")

    # If accessed via GET, redirect authenticated user to dashboard, guest to landing
    if request.user.is_authenticated:
        return redirect("core:dashboard")
    return redirect("core:landing")


# ==============================================================================
# USER PROFILE & SETTINGS
# ==============================================================================

@studio_login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    """View and update profile information, change password, or delete account."""
    user = request.user
    profile, _ = UserProfile.objects.get_or_create(user=user)
    stats = services.get_user_stats(user)

    profile_form = UserProfileForm(user=user, instance=profile)
    password_form = PasswordChangeForm(user=user)

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "update_profile":
            profile_form = UserProfileForm(request.POST, user=user, instance=profile)
            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, "Vos informations de profil ont été mises à jour avec succès.")
                return redirect("accounts:profile")
            else:
                messages.error(request, "Veuillez corriger les erreurs ci-dessous.")

        elif action == "change_password":
            password_form = PasswordChangeForm(user=user, data=request.POST)
            if password_form.is_valid():
                user_updated = password_form.save()
                update_session_auth_hash(request, user_updated)
                messages.success(request, "Votre mot de passe a été modifié avec succès.")
                return redirect("accounts:profile")
            else:
                messages.error(request, "Erreur lors du changement de mot de passe. Veuillez vérifier les critères.")

    context = {
        "title": "Profil & Paramètres",
        "profile": profile,
        "stats": stats,
        "profile_form": profile_form,
        "password_form": password_form,
    }
    return render(request, "accounts/profile.html", context)


@studio_login_required
@require_POST
def delete_account_view(request: HttpRequest) -> HttpResponse:
    """Allow user to irreversibly delete their own account."""
    user = request.user
    password = request.POST.get("confirm_password", "")

    if not user.check_password(password):
        messages.error(request, "Mot de passe incorrect. La suppression du compte a été annulée.")
        return redirect("accounts:profile")

    username = user.username
    logout(request)
    user.delete()
    request.session.flush()

    messages.info(request, f"Le compte de {username} et l'ensemble de ses données ont été supprimés définitivement.")
    return redirect("core:landing")


# ==============================================================================
# HTMX LIVE FIELD VALIDATION
# ==============================================================================

@require_http_methods(["GET", "POST"])
def check_username_view(request: HttpRequest) -> HttpResponse:
    """HTMX endpoint to check username uniqueness in real-time."""
    username = (request.POST.get("username") or request.GET.get("username") or "").strip()
    if not username:
        return HttpResponse("")

    if len(username) < 3:
        return HttpResponse(
            '<div class="flex items-center space-x-1.5 text-xs text-amber-400 mt-1">'
            '<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>'
            '<span>Minimum 3 caractères</span></div>'
        )

    exists = User.objects.filter(username__iexact=username).exists()
    if exists:
        return HttpResponse(
            '<div class="flex items-center space-x-1.5 text-xs text-rose-400 mt-1">'
            '<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>'
            '<span>Ce nom d\'utilisateur est déjà pris</span></div>'
        )

    return HttpResponse(
        '<div class="flex items-center space-x-1.5 text-xs text-emerald-400 mt-1">'
        '<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'
        '<span>Nom d\'utilisateur disponible</span></div>'
    )


@require_http_methods(["GET", "POST"])
def check_email_view(request: HttpRequest) -> HttpResponse:
    """HTMX endpoint to check email uniqueness in real-time."""
    email = (request.POST.get("email") or request.GET.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return HttpResponse("")

    exists = User.objects.filter(email__iexact=email).exists()
    if exists:
        return HttpResponse(
            '<div class="flex items-center space-x-1.5 text-xs text-rose-400 mt-1">'
            '<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>'
            '<span>Cet email est déjà associé à un compte</span></div>'
        )

    return HttpResponse(
        '<div class="flex items-center space-x-1.5 text-xs text-emerald-400 mt-1">'
        '<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'
        '<span>Adresse email disponible</span></div>'
    )


# ==============================================================================
# SAAS ADMIN USER MANAGEMENT
# ==============================================================================

@admin_required
def admin_users_view(request: HttpRequest) -> HttpResponse:
    """Main administrative dashboard to oversee users, roles, and platform activity."""
    users = services.admin_filter_users()
    create_form = AdminCreateUserForm()

    kpis = {
        "total_users": User.objects.count(),
        "active_users": User.objects.filter(is_active=True).count(),
        "admin_count": User.objects.filter(is_staff=True).count(),
        "total_datasets": Dataset.objects.count(),
        "total_models": TrainedModel.objects.count(),
    }

    context = {
        "title": "Gestion des Utilisateurs",
        "users": users,
        "kpis": kpis,
        "create_form": create_form,
    }
    return render(request, "accounts/admin/user_management.html", context)


@admin_required
def admin_users_table_partial(request: HttpRequest) -> HttpResponse:
    """HTMX partial returning the users table filtered by search query, role, and status."""
    q = request.GET.get("q", "")
    role = request.GET.get("role", "")
    status = request.GET.get("status", "")

    users = services.admin_filter_users(query=q, role_filter=role, status_filter=status)
    return render(request, "accounts/admin/partials/_users_table.html", {"users": users})


@admin_required
@require_POST
def admin_toggle_role_view(request: HttpRequest, user_id: int) -> HttpResponse:
    """Toggle role between Administrator and Member."""
    target_user = get_object_or_404(User, pk=user_id)
    if target_user == request.user:
        messages.warning(request, "Vous ne pouvez pas modifier votre propre rôle administrateur.")
        return admin_users_table_partial(request)

    new_role = services.admin_toggle_user_role(target_user)
    messages.success(request, f"Le rôle de {target_user.username} est désormais : {new_role}.")
    return admin_users_table_partial(request)


@admin_required
@require_POST
def admin_toggle_status_view(request: HttpRequest, user_id: int) -> HttpResponse:
    """Toggle user active / inactive status."""
    target_user = get_object_or_404(User, pk=user_id)
    if target_user == request.user:
        messages.warning(request, "Vous ne pouvez pas désactiver votre propre compte.")
        return admin_users_table_partial(request)

    is_active = services.admin_toggle_user_status(target_user)
    status_label = "activé" if is_active else "désactivé / suspendu"
    messages.info(request, f"Le compte de {target_user.username} a été {status_label}.")
    return admin_users_table_partial(request)


@admin_required
@require_POST
def admin_create_user_view(request: HttpRequest) -> HttpResponse:
    """Handle quick user creation from the Admin SaaS drawer/modal."""
    form = AdminCreateUserForm(request.POST)
    if form.is_valid():
        new_user = services.admin_create_user(form.cleaned_data)
        messages.success(request, f"L'utilisateur {new_user.username} a été créé avec succès.")
        return redirect("accounts:admin_users")
    else:
        # Return main page with form errors
        users = services.admin_filter_users()
        kpis = {
            "total_users": User.objects.count(),
            "active_users": User.objects.filter(is_active=True).count(),
            "admin_count": User.objects.filter(is_staff=True).count(),
            "total_datasets": Dataset.objects.count(),
            "total_models": TrainedModel.objects.count(),
        }
        return render(
            request,
            "accounts/admin/user_management.html",
            {"title": "Gestion des Utilisateurs", "users": users, "kpis": kpis, "create_form": form, "modal_open": True},
        )


@admin_required
@require_POST
def admin_delete_user_view(request: HttpRequest, user_id: int) -> HttpResponse:
    """Delete a user account from the admin dashboard."""
    target_user = get_object_or_404(User, pk=user_id)
    if target_user == request.user:
        messages.error(request, "Vous ne pouvez pas supprimer votre propre compte depuis ce panneau.")
        return admin_users_table_partial(request)

    username = target_user.username
    target_user.delete()
    messages.success(request, f"L'utilisateur {username} et ses données ont été supprimés.")
    return admin_users_table_partial(request)

