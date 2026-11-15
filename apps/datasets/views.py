from django.shortcuts import render

def index_view(request):
    return render(request, "datasets/index.html", {
        "title": "Ingestion de Données",
    })

