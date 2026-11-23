# 📂 Dossier d'Architecture & Planification : Authentification & Acteurs (RBAC)

> **Projet :** ML Studio — Master Machine Learning Polyvalent Web Platform  
> **Dossier :** `planning/acteurs_planning/`  
> **Cadre technique :** **100% Python / Django 5.x**, Tailwind CSS, HTMX, Alpine.js (Zéro Node.js).  
> **Statut :** Spécifications complètes & Prêt pour l'implémentation.

---

## 🧭 Sommaire & Organisation du Dossier

Ce dossier décompose et structure l'ensemble des tâches nécessaires à l'implémentation du système d'authentification, de gestion des acteurs et de liaison avec la base de données :

```text
planning/acteurs_planning/
├── README.md                                  # 📍 Le présent document d'orientation globale
├── 00_overview_and_architecture.md            # 🏛️ Architecture globale, vision SaaS & acteurs
├── 01_auth_pages_and_flows.md                 # 🔑 Spécification des flux : /login, /register, /logout, /profile
├── 02_database_integration_and_models.md      # 🗄️ Liaison 100% DB : UserProfile, Dataset FK, TrainedModel
├── 03_roles_permissions_and_rbac.md          # 🛡️ Matrice RBAC, décorateurs de sécurité & KPIs Admin
├── 04_frontend_ui_and_components.md           # 🎨 Spécifications UI (Templates, Sidebar, Topbar, Modale)
└── 05_task_breakdown_and_checklist.md         # 📋 Checklist opérationnelle découpée en 6 Sprints
```

---

## 🎯 Synthèse des 3 Acteurs

| Acteur | Statut Django | Accès & Rôle | Isolation des Données |
| :--- | :--- | :--- | :--- |
| **👁️ Visiteur Anonyme (Guest)** | `is_authenticated = False` | Landing page (`/`), Connexion (`/auth/login/`), Inscription (`/auth/register/`), Reset password. | Aucun accès aux données ni au studio. |
| **👤 Utilisateur Standard (User)**| `is_authenticated = True`<br>`is_staff = False` | Accès complet aux 6 étapes du studio ML, gestion de son profil, déconnexion sécurisée. | **Multi-Tenancy strict** : ne voit et n'édite que ses propres datasets et modèles. |
| **🛡️ Administrateur (Admin)** | `is_staff = True`<br>`is_superuser = True` | Tout le studio + Panneau `/admin/` Django + Dashboard de supervision avec KPIs globaux. | Accès en lecture et modération sur tous les projets de la plateforme. |

---

## 🔗 Couverture des Exigences Spécifiques

1. **Pages d'Authentification :**
   - **`/auth/login/`** : Connexion par nom d'utilisateur ou email, toggle mot de passe (Alpine.js), "Se souvenir de moi" et redirection intelligente avec paramètre `?next=`.
   - **`/auth/register/`** : Inscription avec validation instantanée (HTMX) de l'unicité du nom d'utilisateur et de l'email, jauge de sécurité du mot de passe.
   - **`/auth/logout/`** : Déconnexion sécurisée par méthode HTTP **POST**, intégrée avec une modale de confirmation Alpine.js dans la **Sidebar** et la **Topbar**, avec purge complète de la session (`request.session.flush()`).
   - **`/auth/profile/`** : Gestion des informations personnelles, statistiques de l'utilisateur, changement de mot de passe et zone de danger.

2. **Liaison 100% Base de Données (DB-Linked) :**
   - **`UserProfile`** : Table liée en relation 1-à-1 avec l'utilisateur pour stocker avatar, rôle et préférences.
   - **`Dataset`** : Évolution du modèle avec `user = models.ForeignKey(User, null=True, on_delete=models.CASCADE)`. Les jeux de données d'exemples (Iris, Titanic, etc.) restent partagés avec `user=None`.
   - **`TrainedModel`** : Nouvelle table de persistance reliant chaque modèle entraîné à un utilisateur et à un dataset, enregistrant les hyperparamètres, les scores d'évaluation et le chemin du fichier sérialisé `.joblib`.
   - **`UserActivityLog`** : Audit en base des actions critiques (connexion, upload, entraînement, export).

3. **Respect du Frontend Existant :**
   - Design épuré "Linear / Notion / Omakase" respectant les variables de `templates/base.html` (`#111010` fond, `#1B1918` cartes, `#B8323C` accent Ume rouge).
   - Zéro dépendance externe Node.js/npm. Tout fonctionne via DTL, Tailwind CDN, HTMX CDN et Alpine.js CDN.

---

## 🚀 Prochaine Étape Recommandée

Consultez le fichier [05_task_breakdown_and_checklist.md](file:///home/yaasin-ayadi/Master/Python/projet_ML/planning/acteurs_planning/05_task_breakdown_and_checklist.md) pour débuter l'exécution du **Sprint 1** (création de l'application `apps/accounts` et migration des modèles de base de données).

