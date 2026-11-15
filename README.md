# llm_optimize

Réparation ciblée de LLM à partir des échecs d'évaluation.

Pendant une évaluation, on pose des questions au modèle et on score ses réponses
pour révéler ses points faibles. Si l'on veut ensuite améliorer le modèle en
modifiant ses neurons (dans l'esprit de [Neuroscope](https://neuroscope.io)),
relancer toute la suite d'évaluation est coûteux. `llm_optimize` réduit ce coût
en ne travaillant **que** sur l'utile :

1. il ne teste/analyse que les **questions qui mettent le modèle en défaut** ;
2. il n'extrait que les **paramètres réellement actifs** sur ces échecs
   (neurons différentiellement actifs vs. baseline) ;
3. il produit un **rapport** et des **propositions concrètes de modification de
   poids** (`WeightPatch`), validées avant application.

## L'équipe de 4 agents

| Agent | Module | Rôle |
|---|---|---|
| 🪨 **Failure Miner** | `failure_miner.py` | Filtre les résultats d'éval, ne garde que les échecs, les regroupe par catégorie de faiblesse. |
| 🔬 **Activation Tracer** | `activation_tracer.py` | Rejoue les échecs avec hooks sur les couches, compare aux questions réussies, isole les neurons différentiellement actifs. |
| ✏️ **Weight Editor** | `weight_editor.py` | Transforme les neurons suspects en patchs candidats (dampen/boost des poids sortants). Ne touche jamais le modèle live. |
| ✅ **Validator** | `validator.py` | Teste chaque patch sur une **copie** du modèle : taux d'échecs corrigés + taux de régression sur un échantillon de questions qui passaient. N'accepte que les patchs sous le budget de régression. |

L'`Orchestrator` (`orchestrator.py`) enchaîne les 4 agents et produit le `Report`
(markdown inclus).

```
Dataset d'éval ──► Failure Miner ──► échecs uniquement
                                         │
                     Modèle cible (hooks)│
                                         ▼
                                 Activation Tracer
                                         │ neurons différentiellement actifs
                                         ▼
                                 Weight Editor ──► patchs candidats (ΔW)
                                         │
                                         ▼
                                 Validator ──► rapport final
                                 (échecs corrigés, régressions, verdict)
```

## Installation

```bash
pip install -e .            # cœur (numpy uniquement)
pip install -e ".[hf]"      # + torch/transformers pour un vrai LLM HF
pip install -e ".[dev]"     # + pytest
```

## Utilisation

### 1. À partir de résultats d'évaluation existants (recommandé)

```python
from llm_optimize import Orchestrator, TransformersAdapter

model = TransformersAdapter("EleutherAI/pythia-70m")

# questions: liste de Question(id, text, expected_answer, category)
# results:   liste d'EvalResult(question_id, model_answer, expected_answer,
#                               passed, score, category)
report = Orchestrator(model).run(questions, results, top_k_neurons=20)

print(report.to_markdown())          # rapport complet
for v in report.accepted_patches:    # patchs validés, prêts à appliquer
    print(v.patch.describe())
```

Ou directement depuis des fichiers JSONL (voir `examples/`) :

```python
failed, categories, total = FailureMiner().mine_jsonl(
    "questions.jsonl", "results.jsonl"
)
```

Formats attendus :

```jsonc
// questions.jsonl
{"id": "q1", "text": "What is 2+2?", "expected_answer": "4", "category": "math"}
// results.jsonl
{"question_id": "q1", "model_answer": "5", "expected_answer": "4", "passed": false, "score": 0.0}
```

### 2. Démonstration sans GPU ni gros modèle

```bash
python examples/run_mock.py
```

Un `MockModel` (petit MLP numpy) simule un modèle avec un neuron « quirk »
hyper-activé sur certaines questions : le pipeline le détecte, propose un
dampening de ses poids sortants, le valide (100 % d'échecs corrigés, 0 %
de régression) et rejette les patchs bruités.

## Algorithme statistique d'optimisation des poids (v0.2)

Le chemin par défaut (`WeightEditor` + `Validator`) propose des patchs à échelle
fixe (α = 0.3). Pour un finetuning plus fin, `GreedyPatchOptimizer` améliore
l'algorithme sur trois plans :

1. **Classement statistique des candidats** (`stats.py`) : au lieu du simple
   ratio de moyennes, chaque neuron est noté par **delta de Cliff** (taille
   d'effet robuste entre activations des échecs et de la baseline, par
   question) × **spécificité** (parmi les questions où le neuron est le plus
   actif, quelle fraction sont des échecs). Un neuron qui feu aussi sur les
   questions réussies est rétrogradé avant toute recherche coûteuse.
2. **Recherche d'échelle avec raffinement (line search + zoom)** : l'ampleur
   de la correction n'est plus devinée. Une grille grossière d'échelles est
   évaluée sur une copie du modèle, puis **raffinée itérativement** autour du
   meilleur candidat (zoom ×0.9…×1.1), sous l'objectif
   `J = fix_rate − λ·régressions − μ·|1 − scale|` :
   on récompense la réparation, on punit lourdement les régressions (λ = 5)
   et on préfère l'édition la moins invasive à efficacité égale (μ = 0.05).
   Sur la démo, le raffinement trouve ×0.84 là où la grille fixe s'arrêtait
   à ×0.8 — même réparation, intervention plus douce.
3. **Calibration rank-one (inspirée de ROME)** : quand aucune échelle
   mono-neuron ne suffit (défaillance conjointe de plusieurs neurons, déficit
   absolu…), l'optimiseur propose une édition rank-one de la couche :
   `ΔW = −γ · (sortie_moyenne_échecs − sortie_moyenne_baseline) · k*ᵀ / (k*ᵀk*)`
   où `k*` est l'entrée moyenne de la couche sur les questions échouées.
   C'est la correction de norme minimale qui ramène la réponse de la couche
   au niveau baseline sur la clé d'échec. Un coût fixe (ρ = 0.2) la pénalise
   légèrement : les éditions par échelle, plus interprétables, restent
   prioritaires quand elles suffisent.
4. **Composition gloutonne** : les candidats des deux types (échelle et
   rank-one) sont accumulés dans une édition combinée ; une addition n'est
   conservée que si elle améliore strictement l'objectif global. Les
   interactions entre neurons sont prises en compte et le livrable est
   **une** édition de poids finale, re-validée de bout en bout.

```python
from llm_optimize import GreedyPatchOptimizer, Orchestrator, Validator

report = Orchestrator(
    model,
    optimizer=GreedyPatchOptimizer(validator=Validator()),
).run(questions, results, top_k_neurons=20, trace_method="effect")

print(report.optimization.combined_fix_rate)     # ex. 1.0
print(report.optimization.combined_patches)      # l'édition finale recommandée
```

## Types de patchs proposés

| Type | Condition | Édition |
|---|---|---|
| `dampen` | neuron sur-activé sur les échecs (ratio ≥ 2×) | `W_out[i, :] *= (1 - α)` |
| `boost` | neuron sous-activé (ratio ≤ 0.5×) | `W_out[i, :] *= (1 + β)` |

Sur un vrai modèle HF, `TransformersAdapter.apply_scale` édite la ligne
correspondante de `mlp.down_proj`, c'est-à-dire exactement les poids sortants
d'un neuron intermédiaire. Les seuils, α, β, le budget de régression et le
nombre max de patchs sont configurables dans `WeightEditor` et `Validator`.

## Garde-fous

- Le modèle live n'est **jamais** muté : chaque patch est testé sur une copie.
- Tout patch est accepté seulement si `fix_rate ≥ min_fix_rate` **et**
  `regression_rate ≤ max_regression_rate` (défaut : 50 % / 2 %).
- Le rapport distingue `accepté / rejeté` et documente chaque patch
  (preuves d'activation, catégories de faiblesse, taux mesurés).

## Limites connues

- L'édition de poids est une intervention délicate : à réserver aux modèles
  open-weights dont on maîtrise le déploiement, et à toujours re-valider sur la
  suite d'éval complète avant toute mise en production.
- `TransformersAdapter` cible les architectures causales standard (GPT-2,
  Pythia, OLMo, type Llama) ; l'analyse est mean-poolée sur les tokens.

## Licence

[MIT](LICENSE)
