from django.shortcuts import render

def index_view(request):
    return render(request, "preprocessing/index.html", {
        "title": "Prétraitement & Feature Engineering",
    })

