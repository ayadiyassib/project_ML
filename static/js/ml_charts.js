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

    function isLightMode() {
        return document.documentElement.classList.contains("light") || !document.documentElement.classList.contains("dark");
    }

    function getThemeLayout(isLight) {
        if (isLight) {
            return {
                "font.color": "#334155",
                "hoverlabel.bgcolor": "#ffffff",
                "hoverlabel.bordercolor": "#e2e8f0",
                "hoverlabel.font.color": "#0f172a",
                "legend.font.color": "#475569",
                "xaxis.gridcolor": "rgba(0, 0, 0, 0.07)",
                "xaxis.linecolor": "rgba(0, 0, 0, 0.15)",
                "xaxis.tickcolor": "rgba(0, 0, 0, 0.15)",
                "xaxis.tickfont.color": "#64748b",
                "xaxis.title.font.color": "#334155",
                "yaxis.gridcolor": "rgba(0, 0, 0, 0.07)",
                "yaxis.linecolor": "rgba(0, 0, 0, 0.15)",
                "yaxis.tickcolor": "rgba(0, 0, 0, 0.15)",
                "yaxis.tickfont.color": "#64748b",
                "yaxis.title.font.color": "#334155"
            };
        } else {
            return {
                "font.color": "#9B9388",
                "hoverlabel.bgcolor": "#1b1918",
                "hoverlabel.bordercolor": "#2f2c2a",
                "hoverlabel.font.color": "#F3EDE2",
                "legend.font.color": "#9B9388",
                "xaxis.gridcolor": "rgba(255, 255, 255, 0.06)",
                "xaxis.linecolor": "rgba(255, 255, 255, 0.12)",
                "xaxis.tickcolor": "rgba(255, 255, 255, 0.12)",
                "xaxis.tickfont.color": "#9B9388",
                "xaxis.title.font.color": "#F3EDE2",
                "yaxis.gridcolor": "rgba(255, 255, 255, 0.06)",
                "yaxis.linecolor": "rgba(255, 255, 255, 0.12)",
                "yaxis.tickcolor": "rgba(255, 255, 255, 0.12)",
                "yaxis.tickfont.color": "#9B9388",
                "yaxis.title.font.color": "#F3EDE2"
            };
        }
    }

    function renderOne(el) {
        var source = document.getElementById(el.dataset.plotly);
        if (!source || !window.Plotly) { return; }
        try {
            var fig = JSON.parse(source.textContent);
            el.dataset.rendered = "1";
            var layout = fig.layout || {};
            if (isLightMode()) {
                layout.font = layout.font || {};
                layout.font.color = "#334155";
                if (layout.hoverlabel) {
                    layout.hoverlabel.bgcolor = "#ffffff";
                    layout.hoverlabel.bordercolor = "#e2e8f0";
                    if (layout.hoverlabel.font) layout.hoverlabel.font.color = "#0f172a";
                }
                if (layout.xaxis) {
                    layout.xaxis.gridcolor = "rgba(0, 0, 0, 0.07)";
                    layout.xaxis.linecolor = "rgba(0, 0, 0, 0.15)";
                    layout.xaxis.tickcolor = "rgba(0, 0, 0, 0.15)";
                    if (layout.xaxis.tickfont) layout.xaxis.tickfont.color = "#64748b";
                    if (layout.xaxis.title && layout.xaxis.title.font) layout.xaxis.title.font.color = "#334155";
                }
                if (layout.yaxis) {
                    layout.yaxis.gridcolor = "rgba(0, 0, 0, 0.07)";
                    layout.yaxis.linecolor = "rgba(0, 0, 0, 0.15)";
                    layout.yaxis.tickcolor = "rgba(0, 0, 0, 0.15)";
                    if (layout.yaxis.tickfont) layout.yaxis.tickfont.color = "#64748b";
                    if (layout.yaxis.title && layout.yaxis.title.font) layout.yaxis.title.font.color = "#334155";
                }
            }
            window.Plotly.newPlot(el, fig.data || [], layout, CONFIG);
            if (observer) { observer.observe(el); }
        } catch (err) {
            el.dataset.rendered = "error";
            el.innerHTML = '<p class="text-xs text-rose-400 p-4">Graphique indisponible.</p>';
            if (window.console) { console.error("ml_charts:", err); }
        }
    }

    function updateChartsTheme() {
        if (!window.Plotly) { return; }
        var isLight = isLightMode();
        var update = getThemeLayout(isLight);
        document.querySelectorAll('[data-plotly][data-rendered="1"]').forEach(function (el) {
            try {
                if (el.layout) {
                    var elUpdate = Object.assign({}, update);
                    Object.keys(el.layout).forEach(function (k) {
                        if (/^[xy]axis\d+$/.test(k)) {
                            elUpdate[k + ".gridcolor"] = isLight ? "rgba(0, 0, 0, 0.07)" : "rgba(255, 255, 255, 0.06)";
                            elUpdate[k + ".linecolor"] = isLight ? "rgba(0, 0, 0, 0.15)" : "rgba(255, 255, 255, 0.12)";
                            elUpdate[k + ".tickcolor"] = isLight ? "rgba(0, 0, 0, 0.15)" : "rgba(255, 255, 255, 0.12)";
                            elUpdate[k + ".tickfont.color"] = isLight ? "#64748b" : "#9B9388";
                            elUpdate[k + ".title.font.color"] = isLight ? "#334155" : "#F3EDE2";
                        }
                    });
                    window.Plotly.relayout(el, elUpdate);
                } else {
                    window.Plotly.relayout(el, update);
                }
            } catch (err) {}
        });
    }

    window.addEventListener("theme-changed", updateChartsTheme);

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
