"""
Views for Module 8: Documentation, Algorithm Theoretical Sheets & Interactive Tutorials.
"""

from django.shortcuts import render, Http404
from apps.docs_tutorials import services


def index_view(request):
    """
    Renders documentation portal overview with list of algorithms and tutorials.
    """
    algorithms = services.get_all_algorithms()
    tutorials = services.get_all_tutorials()
    
    # Categorize algorithms
    categories = {
        "Régression": [a for a in algorithms if a["family"] == "regression"],
        "Classification": [a for a in algorithms if a["family"] in ("classification", "trees", "kernel", "neighborhood")],
        "Clustering & Non-Supervisé": [a for a in algorithms if a["family"] == "clustering"],
        "Deep Learning": [a for a in algorithms if a["family"] == "deep-learning"],
    }

    context = {
        "title": "ML Academy & Documentation",
        "algorithms": algorithms,
        "categories": categories,
        "tutorials": tutorials,
    }
    return render(request, "docs_tutorials/index.html", context)


def algorithm_detail_view(request, slug):
    """
    Renders deep theoretical markdown sheet with KaTeX math for a specific algorithm.
    """
    algorithm = services.get_algorithm_by_slug(slug)
    if not algorithm:
        raise Http404("Algorithme non trouvé.")

    context = {
        "title": f"Documentation | {algorithm['name']}",
        "algorithm": algorithm,
        "all_algorithms": services.get_all_algorithms(),
    }
    return render(request, "docs_tutorials/algorithm_detail.html", context)


def tutorial_detail_view(request, slug):
    """
    Renders guided interactive tutorial walkthrough.
    """
    tutorial = services.get_tutorial_by_slug(slug)
    if not tutorial:
        raise Http404("Tutoriel non trouvé.")

    context = {
        "title": f"Tutoriel | {tutorial['title']}",
        "tutorial": tutorial,
        "all_tutorials": services.get_all_tutorials(),
    }
    return render(request, "docs_tutorials/tutorial_detail.html", context)
