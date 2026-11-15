from django.shortcuts import render

def index_view(request):
    return render(request, "ml_engine/index.html", {
        "title": "Entraînement des Modèles ML",
    })

