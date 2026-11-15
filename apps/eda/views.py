from django.shortcuts import render

def index_view(request):
    return render(request, "eda/index.html", {
        "title": "Analyse Exploratoire des Données (EDA)",
    })

