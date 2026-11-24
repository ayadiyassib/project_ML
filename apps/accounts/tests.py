"""
Comprehensive test suite for Authentication, UserProfile, RBAC permissions,
Multi-Tenancy data isolation, and Admin SaaS user management.
"""

from __future__ import annotations

import io
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse

from apps.accounts.models import UserActivityLog, UserProfile
from apps.datasets.models import Dataset
from apps.ml_engine.models import TrainedModel


class UserProfileSignalsTest(TestCase):
    """Test UserProfile automatic creation and signal synchronization."""

    def test_standard_user_profile_creation(self):
        user = User.objects.create_user(
            username="scientist",
            email="scientist@mlstudio.local",
            password="StrongPassword123!",
            first_name="Ada",
            last_name="Lovelace",
        )
        self.assertTrue(hasattr(user, "profile"))
        self.assertEqual(user.profile.role, UserProfile.RoleChoices.USER)
        self.assertEqual(user.profile.initials, "AL")

    def test_admin_user_profile_creation(self):
        admin = User.objects.create_superuser(
            username="admin_user",
            email="admin@mlstudio.local",
            password="AdminPassword123!",
        )
        self.assertTrue(hasattr(admin, "profile"))
        self.assertEqual(admin.profile.role, UserProfile.RoleChoices.ADMIN)
        self.assertEqual(admin.profile.initials, "AD")

    def test_initials_fallback_to_username(self):
        user = User.objects.create_user(username="turing", password="pwd")
        self.assertEqual(user.profile.initials, "TU")


class AuthenticationFlowsTest(TestCase):
    """Test login, registration, logout and password recovery flows."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="marie_curie",
            email="marie@science.fr",
            password="CuriePassword2026!",
            first_name="Marie",
            last_name="Curie",
        )

    def test_landing_page_accessible_to_guest(self):
        response = self.client.get(reverse("core:landing"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "ML Studio")

    def test_authenticated_user_visiting_landing_redirected_to_dashboard(self):
        self.client.login(username="marie_curie", password="CuriePassword2026!")
        response = self.client.get(reverse("core:landing"))
        self.assertRedirects(response, reverse("core:dashboard"))

    def test_guest_access_to_dashboard_redirects_to_login(self):
        response = self.client.get(reverse("core:dashboard"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login/", response.url)

    def test_login_with_username_success(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "marie_curie", "password": "CuriePassword2026!"},
        )
        self.assertRedirects(response, reverse("core:dashboard"))
        self.assertTrue(
            UserActivityLog.objects.filter(
                user=self.user, action=UserActivityLog.ActionChoices.LOGIN
            ).exists()
        )

    def test_login_with_email_success(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "marie@science.fr", "password": "CuriePassword2026!"},
        )
        self.assertRedirects(response, reverse("core:dashboard"))

    def test_login_with_invalid_credentials_fails(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "marie_curie", "password": "WrongPassword!"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Identifiants invalides")

    def test_register_new_user_success(self):
        response = self.client.post(
            reverse("accounts:register"),
            {
                "username": "new_researcher",
                "email": "new@research.org",
                "first_name": "Isaac",
                "last_name": "Newton",
                "password": "GravityPassword99!",
                "password_confirm": "GravityPassword99!",
            },
        )
        self.assertRedirects(response, reverse("core:dashboard"))
        new_user = User.objects.get(username="new_researcher")
        self.assertEqual(new_user.first_name, "Isaac")
        self.assertEqual(new_user.profile.role, UserProfile.RoleChoices.USER)

    def test_logout_via_get_does_not_logout(self):
        self.client.login(username="marie_curie", password="CuriePassword2026!")
        response = self.client.get(reverse("accounts:logout"))
        # Authenticated user GET /auth/logout/ redirects to dashboard without logging out
        self.assertRedirects(response, reverse("core:dashboard"))

    def test_logout_via_post_flushes_session(self):
        self.client.login(username="marie_curie", password="CuriePassword2026!")
        response = self.client.post(reverse("accounts:logout"))
        self.assertRedirects(response, reverse("core:landing"))
        # Verify user is logged out
        response_dash = self.client.get(reverse("core:dashboard"))
        self.assertEqual(response_dash.status_code, 302)


class HTMXValidationTest(TestCase):
    """Test real-time validation endpoints for username and email."""

    def setUp(self):
        self.client = Client()
        User.objects.create_user(
            username="existing_user",
            email="existing@domain.com",
            password="pwd",
        )

    def test_username_available(self):
        response = self.client.post(
            reverse("accounts:check_username"), {"username": "brand_new_user"}
        )
        self.assertIn("Nom d'utilisateur disponible", response.content.decode())

    def test_username_taken(self):
        response = self.client.post(
            reverse("accounts:check_username"), {"username": "existing_user"}
        )
        self.assertContains(response, "déjà pris")

    def test_username_too_short(self):
        response = self.client.post(
            reverse("accounts:check_username"), {"username": "ab"}
        )
        self.assertContains(response, "Minimum 3 caractères")

    def test_email_available(self):
        response = self.client.post(
            reverse("accounts:check_email"), {"email": "free@domain.com"}
        )
        self.assertContains(response, "Adresse email disponible")

    def test_email_taken(self):
        response = self.client.post(
            reverse("accounts:check_email"), {"email": "existing@domain.com"}
        )
        self.assertContains(response, "déjà associé à un compte")


class MultiTenancyAndRBACTest(TestCase):
    """Test data isolation between users and admin privileges."""

    def setUp(self):
        self.client_user1 = Client()
        self.user1 = User.objects.create_user(
            username="user_one", email="user1@test.com", password="Password123!"
        )
        self.client_user1.login(username="user_one", password="Password123!")

        self.client_user2 = Client()
        self.user2 = User.objects.create_user(
            username="user_two", email="user2@test.com", password="Password123!"
        )
        self.client_user2.login(username="user_two", password="Password123!")

        self.client_admin = Client()
        self.admin = User.objects.create_superuser(
            username="boss_admin", email="admin@test.com", password="AdminPassword123!"
        )
        self.client_admin.login(username="boss_admin", password="AdminPassword123!")

        # Create dataset owned by user1
        csv_file = SimpleUploadedFile("user1_data.csv", b"a,b,c\n1,2,3\n4,5,6\n", content_type="text/csv")
        self.dataset_user1 = Dataset.objects.create(
            user=self.user1,
            name="User1 Dataset",
            original_filename="user1_data.csv",
            file=csv_file,
            source=Dataset.Source.UPLOAD,
            n_rows=2,
            n_cols=3,
        )

    def test_user2_cannot_see_user1_dataset(self):
        response = self.client_user2.get(reverse("datasets:index"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "User1 Dataset")

    def test_admin_can_see_user1_dataset(self):
        response = self.client_admin.get(reverse("datasets:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "User1 Dataset")

    def test_user2_cannot_activate_user1_dataset(self):
        response = self.client_user2.post(
            reverse("datasets:activate", kwargs={"pk": self.dataset_user1.pk})
        )
        self.assertEqual(response.status_code, 404)

    def test_user2_cannot_delete_user1_dataset(self):
        response = self.client_user2.post(
            reverse("datasets:delete"),
            {"dataset_id": str(self.dataset_user1.pk)},
        )
        # Should be redirected with error message and file still exists
        self.assertTrue(Dataset.objects.filter(pk=self.dataset_user1.pk).exists())

    def test_standard_user_cannot_access_admin_users(self):
        response = self.client_user1.get(reverse("accounts:admin_users"))
        self.assertEqual(response.status_code, 403)

    def test_admin_can_access_admin_users(self):
        response = self.client_admin.get(reverse("accounts:admin_users"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Gestion des Utilisateurs")
        self.assertContains(response, "user_one")
        self.assertContains(response, "user_two")

    def test_admin_can_toggle_user_role(self):
        self.assertFalse(self.user1.is_staff)
        response = self.client_admin.post(
            reverse("accounts:admin_toggle_role", kwargs={"user_id": self.user1.pk})
        )
        self.assertEqual(response.status_code, 200)
        self.user1.refresh_from_db()
        self.assertTrue(self.user1.is_staff)

    def test_admin_can_toggle_user_status(self):
        self.assertTrue(self.user1.is_active)
        response = self.client_admin.post(
            reverse("accounts:admin_toggle_status", kwargs={"user_id": self.user1.pk})
        )
        self.assertEqual(response.status_code, 200)
        self.user1.refresh_from_db()
        self.assertFalse(self.user1.is_active)

    def test_admin_cannot_alter_own_role(self):
        response = self.client_admin.post(
            reverse("accounts:admin_toggle_role", kwargs={"user_id": self.admin.pk})
        )
        self.admin.refresh_from_db()
        self.assertTrue(self.admin.is_staff)


class ProfileAndUpdateTest(TestCase):
    """Test user profile page and personal info modification."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="ada",
            email="ada@analytics.com",
            password="AdaPassword2026!",
            first_name="Ada",
            last_name="Byron",
        )
        self.client.login(username="ada", password="AdaPassword2026!")

    def test_profile_page_displays_stats(self):
        response = self.client.get(reverse("accounts:profile"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Datasets Importés")
        self.assertContains(response, "Modèles Entraînés")
        self.assertContains(response, "Stockage Données")

    def test_update_profile_info(self):
        response = self.client.post(
            reverse("accounts:profile"),
            {
                "action": "update_profile",
                "first_name": "Augusta",
                "last_name": "King",
                "email": "augusta@analytics.com",
                "organization": "Imperial College",
                "avatar_color": "#B8323C",
            },
        )
        self.assertRedirects(response, reverse("accounts:profile"))
        self.user.refresh_from_db()
        self.assertEqual(self.user.first_name, "Augusta")
        self.assertEqual(self.user.email, "augusta@analytics.com")
        self.assertEqual(self.user.profile.organization, "Imperial College")

