from django.shortcuts import render

def index_view(request):
    return render(request, "model_viz/index.html", {
        "title": "Visualisations Diagnostiques",
    })

