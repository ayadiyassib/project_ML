# 🎨 Spécifications Frontend, Templates & Composants UI

> **Emplacement :** `planning/acteurs_planning/04_frontend_ui_and_components.md`  
> **Conformité stricte :** 100% Python, Django Template Language (DTL), Tailwind CSS, HTMX, Alpine.js (Zéro Node.js/npm).  
> **Identité visuelle :** Palette "Omakase / Linear" (`#111010` fond laque, `#1B1918` carte sumi, `#B8323C` accent Ume, `#F3EDE2` texte papier).

---

## 1. Philosophie & Cohérence Graphique

Les pages et composants d'authentification s'intègrent organiquement dans l'expérience utilisateur existante de ML Studio :

| Élément | Classes Tailwind & Tokens CSS | Rôle Visuel |
| :--- | :--- | :--- |
| **Fond de page** | `bg-[#111010] text-[#F3EDE2]` | Noir laqué profond, anti-éblouissement |
| **Cartes & Conteneurs** | `bg-[#1B1918] border border-[#2f2c2a] rounded-xl shadow-xl` | Effet papier japonais / Linear card |
| **Champs de formulaire** | `bg-[#141312] border border-[#2f2c2a] focus:border-[#B8323C] text-sm` | Saisie contrastée et sans distraction |
| **Bouton Primaire (CTA)**| `bg-[#B8323C] hover:bg-[#98262f] text-white font-medium shadow-md` | Rouge Ume identitaire pour l'action principale |
| **Bouton Secondaire** | `bg-slate-800/60 hover:bg-slate-800 text-slate-300 border border-slate-700/60` | Actions neutres (Annuler, Retour) |
| **Badges de Rôle** | `Admin`: `bg-rose-500/15 text-rose-300 border-rose-500/30`<br>`Membre`: `bg-indigo-500/15 text-indigo-300 border-indigo-500/30` | Identification instantanée du rôle |

---

## 2. Nouveaux Templates d'Authentification

### 2.1 Layout Dédié aux Visiteurs — `templates/layouts/auth_base.html`
Ce layout est spécialement conçu pour `/auth/login/`, `/auth/register/` et `/auth/password-reset/`. Il élimine la barre latérale et le stepper horizontal pour offrir un tunnel de conversion centré et immersif :

```html
<!DOCTYPE html>
<html lang="fr" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Authentification | ML Studio{% endblock %}</title>
    
    <!-- Polices Google : Shippori Mincho & Noto Sans JP -->
    <link href="https://fonts.googleapis.com/css2?family=Shippori+Mincho:wght@500;700&family=Noto+Sans+JP:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- Tailwind CSS (Configuration Omakase partagée) -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@1.9.12"></script>
    <script defer src="https://unpkg.com/alpinejs@3.13.10/dist/cdn.min.js"></script>
</head>
<body class="bg-[#111010] text-slate-100 min-h-screen flex flex-col justify-between antialiased selection:bg-[#B8323C] selection:text-white">

    <!-- En-tête minimal avec Logo ML Studio -->
    <header class="py-6 px-8 flex items-center justify-between">
        <a href="{% url 'core:landing' %}" class="flex items-center space-x-3">
            <div class="w-9 h-9 rounded-[2px] bg-[#B8323C] flex items-center justify-center shrink-0 shadow-lg">
                <svg class="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
            </div>
            <span class="font-display font-semibold tracking-wide text-paper text-lg">ML Studio</span>
        </a>
        <div class="text-xs text-slate-400">
            {% block header_action %}{% endblock %}
        </div>
    </header>

    <!-- Conteneur centré du formulaire -->
    <main class="flex-1 flex items-center justify-center px-4 py-8">
        <div class="w-full max-w-md bg-[#1B1918] border border-[#2f2c2a] rounded-2xl p-8 shadow-2xl backdrop-blur-sm">
            {% include 'components/toasts.html' %}
            {% block auth_content %}{% endblock %}
        </div>
    </main>

    <!-- Pied de page discret -->
    <footer class="py-4 text-center text-xs text-slate-500">
        &copy; 2026 ML Studio &bull; Plateforme AutoML 100% Python &bull; Conçu pour Master ML
    </footer>
</body>
</html>
```

---

### 2.2 Template de Connexion avec Toggle Password (Alpine.js) — `templates/accounts/login.html`

```html
{% extends 'layouts/auth_base.html' %}

{% block title %}Connexion | ML Studio{% endblock %}

{% block header_action %}
    Pas encore de compte ? 
    <a href="{% url 'accounts:register' %}" class="text-[#B8323C] hover:underline font-medium ml-1">S'inscrire</a>
{% endblock %}

{% block auth_content %}
<div x-data="{ showPassword: false }">
    <div class="mb-6 text-center">
        <h1 class="font-display text-2xl font-bold text-white tracking-tight">Bon retour 👋</h1>
        <p class="text-xs text-slate-400 mt-1">Connectez-vous pour retrouver vos projets et modèles</p>
    </div>

    <form method="post" action="{% url 'accounts:login' %}" class="space-y-4">
        {% csrf_token %}
        <input type="hidden" name="next" value="{{ next }}">

        <!-- Champ Identifiant / Email -->
        <div>
            <label class="block text-xs font-medium text-slate-300 mb-1.5 uppercase tracking-wider">Identifiant ou Email</label>
            <input type="text" name="username" required autofocus
                   class="w-full px-3.5 py-2.5 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:border-[#B8323C] transition-colors"
                   placeholder="nom d'utilisateur ou email">
        </div>

        <!-- Champ Mot de passe avec oeil Alpine.js -->
        <div>
            <div class="flex items-center justify-between mb-1.5">
                <label class="text-xs font-medium text-slate-300 uppercase tracking-wider">Mot de passe</label>
                <a href="{% url 'accounts:password_reset' %}" class="text-xs text-slate-400 hover:text-[#B8323C] transition-colors">Mot de passe oublié ?</a>
            </div>
            <div class="relative">
                <input :type="showPassword ? 'text' : 'password'" name="password" required
                       class="w-full px-3.5 py-2.5 pr-10 rounded-lg bg-[#141312] border border-[#2f2c2a] text-slate-100 text-sm focus:outline-none focus:border-[#B8323C] transition-colors"
                       placeholder="••••••••••••">
                <button type="button" @click="showPassword = !showPassword" 
                        class="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-200">
                    <svg x-show="!showPassword" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                    <svg x-show="showPassword" x-cloak class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l18 18" />
                    </svg>
                </button>
            </div>
        </div>

        <!-- Se souvenir de moi -->
        <div class="flex items-center">
            <input type="checkbox" id="remember_me" name="remember_me" class="w-4 h-4 rounded bg-[#141312] border-[#2f2c2a] text-[#B8323C]">
            <label for="remember_me" class="ml-2 text-xs text-slate-300">Se souvenir de moi pendant 30 jours</label>
        </div>

        <!-- Bouton de Soumission -->
        <button type="submit" 
                class="w-full py-2.5 px-4 bg-[#B8323C] hover:bg-[#98262f] text-white font-medium rounded-lg text-sm transition-all duration-200 shadow-lg flex items-center justify-center space-x-2">
            <span>Se connecter au Studio</span>
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
        </button>
    </form>
</div>
{% endblock %}
```

---

## 3. Évolution de la Navigation Interne (Sidebar & Topbar)

### 3.1 Intégration de la Gestion Administrateur dans la Sidebar (`templates/components/sidebar.html`)

Dans la barre latérale, nous ajoutons une section dédiée **"Administration"** positionnée avec élégance entre "Pipeline Studio" et "Ressources", visible **exclusivement pour les administrateurs** (`request.user.is_staff`) :

```html
<!-- Section Administration réservée à l'Admin dans la Sidebar -->
{% if request.user.is_staff %}
<div x-show="sidebarOpen" class="pt-5 pb-1 px-3 text-[9px] font-semibold tracking-label text-rose-400 uppercase flex items-center justify-between">
    <span>Administration</span>
    <span class="h-1.5 w-1.5 rounded-full bg-rose-500 animate-pulse"></span>
</div>

<!-- Option 1 : Gestion des Utilisateurs -->
<a href="{% url 'accounts:admin_users' %}" 
   class="flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors {% if 'admin_users' in request.resolver_match.url_name %}bg-rose-500/10 text-rose-300 border border-rose-500/25{% else %}text-slate-400 hover:text-slate-200 hover:bg-slate-800/60{% endif %}">
    <svg class="w-5 h-5 shrink-0 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
    </svg>
    <span x-show="sidebarOpen" x-transition.opacity class="truncate">Gestion Utilisateurs</span>
    <span x-show="sidebarOpen" class="ml-auto px-1.5 py-0.5 text-[9px] bg-rose-950/80 border border-rose-500/30 text-rose-300 rounded font-mono">Admin</span>
</a>

<!-- Option 2 : Django Admin Rapide -->
<a href="{% url 'admin:index' %}" target="_blank"
   class="flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-amber-300 hover:bg-slate-800/60 transition-colors">
    <svg class="w-5 h-5 shrink-0 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    </svg>
    <span x-show="sidebarOpen" x-transition.opacity class="truncate">Panneau /admin</span>
</a>
{% endif %}
```

---

### 3.2 Intégration de l'Utilisateur dans la Sidebar (`templates/components/sidebar.html`)
Remplacement du simple pied de page textuel par la carte utilisateur interactive :

```html
<!-- Carte Utilisateur en Pied de Sidebar -->
<div class="p-3 border-t border-slate-800/80 bg-[#141312]">
    <div class="flex items-center justify-between">
        
        <!-- Identité & Avatar -->
        <a href="{% url 'accounts:profile' %}" class="flex items-center space-x-3 overflow-hidden group">
            <div class="w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs text-white shrink-0 shadow"
                 style="background-color: {{ request.user.profile.avatar_color|default:'#B8323C' }}">
                {{ request.user.profile.initials|default:"ML" }}
            </div>
            <div x-show="sidebarOpen" x-transition.opacity class="flex flex-col min-w-0">
                <span class="text-xs font-semibold text-slate-200 truncate group-hover:text-white transition-colors">
                    {{ request.user.get_full_name|default:request.user.username }}
                </span>
                <span class="text-[9px] font-mono tracking-wider uppercase {% if request.user.is_staff %}text-rose-400{% else %}text-indigo-400{% endif %}">
                    {% if request.user.is_staff %}Administrateur{% else %}Membre Studio{% endif %}
                </span>
            </div>
        </a>

        <!-- Bouton Déconnexion Rapide -->
        <button type="button" 
                @click="$dispatch('open-logout-modal')"
                title="Se déconnecter"
                class="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-800/80 transition-colors">
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
        </button>
    </div>
</div>
```

---

### 3.2 Menu Déroulant Profil dans la Topbar (`templates/components/topbar.html`)
Ajout d'un menu déroulant à droite dans l'en-tête principal :

```html
<!-- Menu Utilisateur Déroulant Topbar -->
<div class="relative" x-data="{ userMenuOpen: false }">
    <button @click="userMenuOpen = !userMenuOpen" 
            class="flex items-center space-x-2 pl-2 pr-3 py-1 rounded-full bg-slate-900 border border-slate-800 hover:border-slate-700 transition-colors focus:outline-none">
        <div class="w-6 h-6 rounded-full flex items-center justify-center font-bold text-[10px] text-white"
             style="background-color: {{ request.user.profile.avatar_color|default:'#B8323C' }}">
            {{ request.user.profile.initials|default:"ML" }}
        </div>
        <span class="text-xs font-medium text-slate-300 hidden md:inline">{{ request.user.username }}</span>
        <svg class="w-3 h-3 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
    </button>

    <!-- Menu Dropdown -->
    <div x-show="userMenuOpen" 
         @click.outside="userMenuOpen = false"
         x-transition:enter="transition ease-out duration-100"
         x-transition:enter-start="transform opacity-0 scale-95"
         x-transition:enter-end="transform opacity-100 scale-100"
         class="absolute right-0 mt-2 w-56 bg-[#1B1918] border border-[#2f2c2a] rounded-xl shadow-2xl py-1.5 z-50 divide-y divide-slate-800">
        
        <div class="px-4 py-2.5">
            <p class="text-xs font-semibold text-white">{{ request.user.get_full_name|default:request.user.username }}</p>
            <p class="text-[11px] text-slate-400 truncate">{{ request.user.email }}</p>
        </div>

        <div class="py-1">
            <a href="{% url 'accounts:profile' %}" class="flex items-center px-4 py-2 text-xs text-slate-300 hover:text-white hover:bg-slate-800/60">
                <svg class="w-3.5 h-3.5 mr-2 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                Mon Profil & Paramètres
            </a>
            {% if request.user.is_staff %}
            <a href="{% url 'admin:index' %}" target="_blank" class="flex items-center px-4 py-2 text-xs text-amber-300 hover:bg-slate-800/60">
                <svg class="w-3.5 h-3.5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                </svg>
                Administration Django
            </a>
            {% endif %}
        </div>

        <div class="py-1">
            <button type="button" @click="$dispatch('open-logout-modal'); userMenuOpen = false;"
                    class="w-full flex items-center px-4 py-2 text-xs text-rose-400 hover:bg-rose-950/20">
                <svg class="w-3.5 h-3.5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Déconnexion
            </button>
        </div>
    </div>
</div>
```

---

### 3.3 Modale de Déconnexion Alpine.js — `templates/components/logout_modal.html`

Pour sécuriser la déconnexion par un `POST` et éviter les déconnexions involontaires :

```html
<div x-data="{ open: false }" 
     @open-logout-modal.window="open = true"
     x-show="open" 
     x-cloak
     class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
    
    <div @click.outside="open = false" 
         class="w-full max-w-sm bg-[#1B1918] border border-[#2f2c2a] rounded-2xl p-6 shadow-2xl">
        
        <div class="w-10 h-10 rounded-full bg-rose-950/60 border border-rose-600/30 flex items-center justify-center text-rose-400 mb-4">
            <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
        </div>

        <h3 class="text-base font-bold text-white mb-1 font-display">Confirmer la déconnexion</h3>
        <p class="text-xs text-slate-400 mb-6 leading-relaxed">
            Êtes-vous certain de vouloir quitter votre session ? Les pipelines ou calculs non enregistrés seront conservés uniquement si un dataset actif est associé à votre compte.
        </p>

        <form method="post" action="{% url 'accounts:logout' %}" class="flex items-center justify-end space-x-3">
            {% csrf_token %}
            <button type="button" @click="open = false"
                    class="px-4 py-2 rounded-lg text-xs font-medium text-slate-300 hover:bg-slate-800 transition-colors">
                Annuler
            </button>
            <button type="submit"
                    class="px-4 py-2 rounded-lg text-xs font-medium bg-[#B8323C] hover:bg-[#98262f] text-white transition-colors shadow">
                Se déconnecter
            </button>
        </form>
    </div>
</div>
```

---

### 3.4 Template Complet de Gestion des Utilisateurs — `templates/accounts/admin/user_management.html`

Cette vue SaaS permet à l'Admin de piloter tous les utilisateurs directement dans l'application avec recherche HTMX en direct et modales de gestion :

```html
{% extends 'base.html' %}

{% block title %}Gestion des Utilisateurs | ML Studio Admin{% endblock %}

{% block content %}
<div class="space-y-6" x-data="{ createModalOpen: false }">
    
    <!-- En-tête de la Page avec Actions -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
            <div class="flex items-center space-x-2">
                <h1 class="text-2xl font-bold font-display text-white">Gestion des Utilisateurs</h1>
                <span class="px-2 py-0.5 text-xs font-mono rounded bg-rose-500/10 text-rose-300 border border-rose-500/30">Admin SaaS</span>
            </div>
            <p class="text-xs text-slate-400 mt-1">Supervisez les comptes enregistrés, attribuez des rôles et gérez les accès à la plateforme.</p>
        </div>

        <button @click="createModalOpen = true" 
                class="inline-flex items-center space-x-2 px-4 py-2 bg-[#B8323C] hover:bg-[#98262f] text-white text-xs font-medium rounded-lg transition-colors shadow">
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
            <span>Nouvel Utilisateur</span>
        </button>
    </div>

    <!-- Barre de Recherche & Filtres HTMX -->
    <div class="bg-[#1B1918] border border-[#2f2c2a] rounded-xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <!-- Recherche Instantanée -->
        <div class="relative w-full md:w-96">
            <span class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
            </span>
            <input type="text" name="q" 
                   hx-get="{% url 'accounts:admin_users_table' %}" 
                   hx-trigger="keyup changed delay:300ms" 
                   hx-target="#users-table-container"
                   placeholder="Rechercher par nom, username ou email..." 
                   class="w-full pl-9 pr-4 py-2 bg-[#141312] border border-[#2f2c2a] rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-[#B8323C] transition-colors">
        </div>

        <!-- Filtres par Rôle & Statut -->
        <div class="flex items-center space-x-3 w-full md:w-auto">
            <select name="role" 
                    hx-get="{% url 'accounts:admin_users_table' %}" 
                    hx-target="#users-table-container"
                    class="bg-[#141312] border border-[#2f2c2a] text-xs text-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:border-[#B8323C]">
                <option value="">Tous les rôles</option>
                <option value="admin">Administrateurs</option>
                <option value="user">Membres standard</option>
            </select>
            <select name="status" 
                    hx-get="{% url 'accounts:admin_users_table' %}" 
                    hx-target="#users-table-container"
                    class="bg-[#141312] border border-[#2f2c2a] text-xs text-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:border-[#B8323C]">
                <option value="">Tous les statuts</option>
                <option value="active">Actifs</option>
                <option value="inactive">Inactifs / Suspendus</option>
            </select>
        </div>
    </div>

    <!-- Conteneur Tableau des Utilisateurs (Cible HTMX) -->
    <div id="users-table-container" class="bg-[#1B1918] border border-[#2f2c2a] rounded-xl overflow-hidden shadow-xl">
        {% include 'accounts/admin/partials/_users_table.html' %}
    </div>

    <!-- Modale Alpine.js de Création Rapide d'Utilisateur -->
    {% include 'accounts/admin/partials/_create_user_modal.html' %}
</div>
{% endblock %}
```


