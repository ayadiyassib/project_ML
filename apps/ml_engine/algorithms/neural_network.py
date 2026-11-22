"""
Artificial Neural Network algorithm implementation:
11. Multi-Layer Perceptron (MLPClassifier & MLPRegressor)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier, MLPRegressor

from apps.ml_engine.algorithms.base import BaseAlgorithm, HyperparamField


def parse_hidden_layers(raw: str) -> Tuple[int, ...]:
    """Parse a string like '64, 32' or '100' into a tuple of ints."""
    try:
        cleaned = [int(p.strip()) for p in raw.replace("(", "").replace(")", "").split(",") if p.strip()]
        if cleaned and all(c > 0 for c in cleaned):
            return tuple(cleaned)
    except Exception:
        pass
    return (64, 32)


class NeuralNetworkAlgorithm(BaseAlgorithm):
    """Algorithme 11 : Réseaux de Neurones Artificiels (MLP - Multi-Layer Perceptron)."""

    key = "neural_network"
    name = "Réseau de Neurones (MLP)"
    category = "both"
    description = "Perceptron multicouche à propagation avant et rétropropagation du gradient, capable d'apprendre des représentations fortement non linéaires."
    formula_latex = r"\mathbf{a}^{(l)} = \sigma(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}) \quad\text{avec rétropropagation}"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="hidden_layer_sizes",
                label="Architecture des couches cachées",
                field_type="select",
                default="64,32",
                options=[
                    ("50", "1 couche (50 neurones)"),
                    ("100", "1 couche (100 neurones)"),
                    ("64,32", "2 couches (64, 32 neurones) - Recommandé"),
                    ("128,64", "2 couches (128, 64 neurones)"),
                    ("64,32,16", "3 couches profondes (64, 32, 16)"),
                ],
                help_text="Nombre de neurones par couche cachée.",
            ),
            HyperparamField(
                name="activation",
                label="Fonction d'activation",
                field_type="select",
                default="relu",
                options=[
                    ("relu", "ReLU (max(0, z) - Rapide & moderne)"),
                    ("tanh", "Tangente Hyperbolique (tanh - symétrique [-1, 1])"),
                    ("logistic", "Sigmoïde Logistique (1 / (1 + e⁻ᶻ))"),
                ],
                help_text="Non-linéarité appliquée à chaque neurone.",
            ),
            HyperparamField(
                name="solver",
                label="Optimiseur (Solveur)",
                field_type="select",
                default="adam",
                options=[
                    ("adam", "Adam (Gradient stochastique adaptatif - Recommandé)"),
                    ("sgd", "SGD (Descente de gradient stochastique standard)"),
                ],
                help_text="Algorithme d'optimisation des poids synaptiques.",
            ),
            HyperparamField(
                name="alpha",
                label="Régularisation L2 (alpha)",
                field_type="float",
                default=0.0001,
                min_val=1e-6,
                max_val=1.0,
                help_text="Pénalité L2 sur les poids pour contrer le surapprentissage.",
            ),
            HyperparamField(
                name="max_iter",
                label="Nombre maximal d'époques (max_iter)",
                field_type="int",
                default=300,
                min_val=50,
                max_val=2000,
                step=50,
                help_text="Nombre maximal de passes complètes sur les données.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        layers_raw = field_dict["hidden_layer_sizes"].clean_value(params.get("hidden_layer_sizes"))
        hidden_layers = parse_hidden_layers(str(layers_raw))
        activation = field_dict["activation"].clean_value(params.get("activation"))
        solver = field_dict["solver"].clean_value(params.get("solver"))
        alpha = field_dict["alpha"].clean_value(params.get("alpha"))
        max_iter = field_dict["max_iter"].clean_value(params.get("max_iter"))

        if task_type == "classification":
            return MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver=solver,
                alpha=float(alpha),
                max_iter=int(max_iter),
                random_state=42,
                early_stopping=False,
            )
        return MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            alpha=float(alpha),
            max_iter=int(max_iter),
            random_state=42,
            early_stopping=False,
        )

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "neural_network",
            "n_layers": getattr(model, "n_layers_", 0),
            "n_iter": getattr(model, "n_iter_", 0),
            "best_loss": round(float(getattr(model, "best_loss_", 0.0) or 0.0), 5),
        }
        if hasattr(model, "loss_curve_"):
            # Sample down if loss curve has hundreds of points for fast chart rendering
            curve = [round(float(l), 5) for l in model.loss_curve_]
            result["loss_curve"] = curve
            result["final_loss"] = curve[-1] if curve else 0.0

        return result
