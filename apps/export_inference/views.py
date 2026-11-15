from django.shortcuts import render

def index_view(request):
    return render(request, "export_inference/index.html", {
        "title": "Export de Pipeline & Inférence",
    })

