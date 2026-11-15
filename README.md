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
