"""
Forms for authentication, user registration, profile management, and admin user creation.
"""

from __future__ import annotations

from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError

from apps.accounts.models import UserProfile


class LoginForm(forms.Form):
    """Login form accepting either username or email with remember me checkbox."""

    username = forms.CharField(
        label="Identifiant ou Email",
        max_length=150,
        widget=forms.TextInput(
            attrs={
                "class": "w-full px-3.5 py-2.5 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:border-[#B8323C] transition-colors",
                "placeholder": "Nom d'utilisateur ou email",
                "autofocus": True,
            }
        ),
    )
    password = forms.CharField(
        label="Mot de passe",
        widget=forms.PasswordInput(
            attrs={
                "class": "w-full px-3.5 py-2.5 pr-10 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 text-sm focus:outline-none focus:border-[#B8323C] transition-colors",
                "placeholder": "••••••••••••",
            }
        ),
    )
    remember_me = forms.BooleanField(
        label="Se souvenir de moi",
        required=False,
        widget=forms.CheckboxInput(
            attrs={
                "class": "w-4 h-4 rounded bg-[#141312] border-[#2f2c2a] text-[#B8323C] focus:ring-0",
            }
        ),
    )


class RegisterForm(forms.Form):
    """Registration form with live validation and password confirmation."""

    username = forms.CharField(
        label="Nom d'utilisateur",
        max_length=150,
        widget=forms.TextInput(
            attrs={
                "class": "w-full px-3.5 py-2.5 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:border-[#B8323C] transition-colors",
                "placeholder": "yassine_ml",
                "hx-post": "/auth/check-username/",
                "hx-trigger": "keyup changed delay:300ms",
                "hx-target": "#username-feedback",
            }
        ),
    )
    first_name = forms.CharField(
        label="Prénom",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "w-full px-3.5 py-2.5 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:border-[#B8323C] transition-colors",
                "placeholder": "Yassine",
            }
        ),
    )
    last_name = forms.CharField(
        label="Nom",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "w-full px-3.5 py-2.5 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:border-[#B8323C] transition-colors",
                "placeholder": "Ayadi",
            }
        ),
    )
    email = forms.EmailField(
        label="Adresse Email",
        widget=forms.EmailInput(
            attrs={
                "class": "w-full px-3.5 py-2.5 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:border-[#B8323C] transition-colors",
                "placeholder": "nom@example.com",
                "hx-post": "/auth/check-email/",
                "hx-trigger": "keyup changed delay:300ms",
                "hx-target": "#email-feedback",
            }
        ),
    )
    password = forms.CharField(
        label="Mot de passe",
        widget=forms.PasswordInput(
            attrs={
                "class": "w-full px-3.5 py-2.5 pr-10 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 text-sm focus:outline-none focus:border-[#B8323C] transition-colors",
                "placeholder": "••••••••••••",
                "@input": "checkStrength($event.target.value)",
            }
        ),
    )
    password_confirm = forms.CharField(
        label="Confirmer le mot de passe",
        widget=forms.PasswordInput(
            attrs={
                "class": "w-full px-3.5 py-2.5 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 text-sm focus:outline-none focus:border-[#B8323C] transition-colors",
                "placeholder": "••••••••••••",
            }
        ),
    )

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        if len(username) < 3:
            raise ValidationError("Le nom d'utilisateur doit contenir au moins 3 caractères.")
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError("Ce nom d'utilisateur est déjà utilisé.")
        return username

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("Cette adresse email est déjà associée à un compte.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")

        if password and password_confirm:
            if password != password_confirm:
                self.add_error("password_confirm", "Les mots de passe ne correspondent pas.")
            else:
                validate_password(password)
        return cleaned_data


class UserProfileForm(forms.ModelForm):
    """Profile editing form."""

    first_name = forms.CharField(
        label="Prénom",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "w-full px-3.5 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-sm focus:outline-none focus:border-[#B8323C]"}
        ),
    )
    last_name = forms.CharField(
        label="Nom",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "w-full px-3.5 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-sm focus:outline-none focus:border-[#B8323C]"}
        ),
    )
    email = forms.EmailField(
        label="Adresse Email",
        widget=forms.EmailInput(
            attrs={"class": "w-full px-3.5 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-sm focus:outline-none focus:border-[#B8323C]"}
        ),
    )
    organization = forms.CharField(
        label="Organisation / Entreprise",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "w-full px-3.5 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-sm focus:outline-none focus:border-[#B8323C]"}
        ),
    )
    avatar_color = forms.CharField(
        label="Couleur d'avatar",
        max_length=7,
        widget=forms.TextInput(
            attrs={"type": "color", "class": "w-12 h-9 p-1 rounded-lg bg-[#141312] border border-[#2f2c2a] cursor-pointer"}
        ),
    )

    class Meta:
        model = UserProfile
        fields = ["organization", "avatar_color"]

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        if user:
            self.fields["first_name"].initial = user.first_name
            self.fields["last_name"].initial = user.last_name
            self.fields["email"].initial = user.email
            self.user = user

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exclude(pk=self.user.pk).exists():
            raise ValidationError("Cette adresse email est déjà utilisée par un autre compte.")
        return email

    def save(self, commit=True):
        profile = super().save(commit=False)
        if self.user:
            self.user.first_name = self.cleaned_data["first_name"].strip()
            self.user.last_name = self.cleaned_data["last_name"].strip()
            self.user.email = self.cleaned_data["email"].strip().lower()
            if commit:
                self.user.save()
                profile.save()
        return profile


class AdminCreateUserForm(forms.Form):
    """Admin modal form to create a new user."""

    username = forms.CharField(
        label="Nom d'utilisateur",
        max_length=150,
        widget=forms.TextInput(
            attrs={"class": "w-full px-3 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-xs focus:outline-none focus:border-[#B8323C]", "placeholder": "nouvel_utilisateur"}
        ),
    )
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(
            attrs={"class": "w-full px-3 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-xs focus:outline-none focus:border-[#B8323C]", "placeholder": "user@example.com"}
        ),
    )
    first_name = forms.CharField(
        label="Prénom",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "w-full px-3 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-xs focus:outline-none focus:border-[#B8323C]"}
        ),
    )
    last_name = forms.CharField(
        label="Nom",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "w-full px-3 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-xs focus:outline-none focus:border-[#B8323C]"}
        ),
    )
    role = forms.ChoiceField(
        label="Rôle",
        choices=[("USER", "Membre Standard"), ("ADMIN", "Administrateur")],
        widget=forms.Select(
            attrs={"class": "w-full px-3 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-xs focus:outline-none focus:border-[#B8323C]"}
        ),
    )
    password = forms.CharField(
        label="Mot de passe initial",
        widget=forms.PasswordInput(
            attrs={"class": "w-full px-3 py-2 rounded-lg bg-[#141312] border border-[#2f2c2a] text-white text-xs focus:outline-none focus:border-[#B8323C]", "placeholder": "••••••••••••"}
        ),
    )

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError("Ce nom d'utilisateur existe déjà.")
        return username

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("Cet email existe déjà.")
        return email

    def clean_password(self):
        password = self.cleaned_data["password"]
        validate_password(password)
        return password

