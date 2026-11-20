/*
 * ML Studio — Plotly renderer shared by the EDA and diagnostics modules.
 *
 * Django embeds each figure as JSON in <script type="application/json" id="…">
 * next to a <div data-plotly="…">.  This script turns those into live Plotly
 * charts:
 *   - on page load, and after every HTMX swap (htmx:afterSettle);
 *   - only when the container is visible (charts inside a hidden Alpine tab
 *     are rendered when the tab opens, so they get the right width);
 *   - purged before HTMX replaces them (no leaked WebGL contexts);
 *   - resized with their container (sidebar collapse, window resize).
 */
(function () {
    "use strict";

    var CONFIG = {
        responsive: true,
        displaylogo: false,
        scrollZoom: false,
        modeBarButtonsToRemove: ["lasso2d", "autoScale2d", "toggleSpikelines", "hoverCompareCartesian"],
        toImageButtonOptions: { format: "png", scale: 2, filename: "ml-studio-chart" }
    };

    var observer = window.ResizeObserver
        ? new ResizeObserver(function (entries) {
            entries.forEach(function (entry) {
                var el = entry.target;
                if (el.dataset.rendered && el.offsetParent !== null && window.Plotly) {
                    window.Plotly.Plots.resize(el);
                }
            });
        })
        : null;

    function isVisible(el) {
        return el.offsetParent !== null && el.getClientRects().length > 0;
    }

    function renderOne(el) {
        var source = document.getElementById(el.dataset.plotly);
        if (!source || !window.Plotly) { return; }
        try {
            var fig = JSON.parse(source.textContent);
            el.dataset.rendered = "1";
            window.Plotly.newPlot(el, fig.data || [], fig.layout || {}, CONFIG);
            if (observer) { observer.observe(el); }
        } catch (err) {
            el.dataset.rendered = "error";
            el.innerHTML = '<p class="text-xs text-rose-400 p-4">Graphique indisponible.</p>';
            if (window.console) { console.error("ml_charts:", err); }
        }
    }

    function render(root) {
        var scope = root && root.querySelectorAll ? root : document;
        scope.querySelectorAll("[data-plotly]:not([data-rendered])").forEach(function (el) {
            if (isVisible(el)) { renderOne(el); }
        });
        // Charts that were already drawn while hidden need a resize once shown.
        scope.querySelectorAll('[data-plotly][data-rendered="1"]').forEach(function (el) {
            if (isVisible(el) && window.Plotly) { window.Plotly.Plots.resize(el); }
        });
    }

    function purge(root) {
        if (!root || !root.querySelectorAll || !window.Plotly) { return; }
        root.querySelectorAll('[data-plotly][data-rendered="1"]').forEach(function (el) {
            if (observer) { observer.unobserve(el); }
            try { window.Plotly.purge(el); } catch (err) { /* already gone */ }
        });
    }

    document.addEventListener("DOMContentLoaded", function () { render(document); });
    document.addEventListener("htmx:afterSettle", function (event) { render(event.target); });
    document.addEventListener("htmx:beforeSwap", function (event) {
        if (event.detail && event.detail.target) { purge(event.detail.target); }
    });

    // Called by Alpine tab switches: x-show changes visibility without a swap.
    window.mlCharts = {
        render: function (root) { window.requestAnimationFrame(function () { render(root || document); }); }
    };
})();
