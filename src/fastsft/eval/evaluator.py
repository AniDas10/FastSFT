"""Orchestrates one evaluation run over a resolved eval prompt set.

For each prompt it collects three answers -- parent (teacher, via OpenRouter),
tuned child, and untuned child (both local) -- then scores:
  - tuned vs untuned  (primary: did fine-tuning improve quality?)
  - parent-likeness    (distillation objective: is the tuned child more like the
                        parent's style than the untuned child? -- reference-judged)
  - tuned vs parent    (gap to the teacher)
via a pairwise LLM judge, and the embedding similarity of each child to the
parent (distillation fidelity). Each pair is judged in both A/B orders to cancel
the judge's position bias. Produces a plain dict; persistence/interpretation
live in eval/results.py, and prompt-set lifecycle in eval/run.py.
"""

from fastsft.eval.config import EvalConfig
from fastsft.eval.constants import COMPARISON_JUDGE_INSTRUCTION, STYLE_JUDGE_INSTRUCTION
from fastsft.eval.embeddings import pairwise_similarities
from fastsft.eval.inference import ChildInferenceEngine
from fastsft.model.base import Model
from fastsft.model.judge import Judge, Verdict
from fastsft.progress import ProgressLogger, rule

# Quality credited to the first answer for one verdict: a clear win, a tie, a
# loss. Averaged across A/B-swapped orders into a per-prompt score.
_WIN, _TIE, _LOSS = 1.0, 0.5, 0.0

# How many worked examples to keep for qualitative inspection in the viewer.
_SAMPLES_KEPT = 3


class Evaluator(ProgressLogger):
    """Runs the answer-generation + judging + similarity pipeline for one adapter.

    Progress logging (self._log) comes from ProgressLogger -- the same shared
    rich console the pipeline stages log through.
    """

    def __init__(self, config: EvalConfig, verbose: bool = True):
        super().__init__(verbose=verbose)
        self._config = config

    def run(self, prompts: list[str]) -> dict:
        """Validate the input, then run -- mirrors the Stage validate-then-run
        template, and brackets the run with a start/end partition rule so eval
        reads like the pipeline stages regardless of caller (CLI or direct)."""
        self._validate_input(prompts)
        if self._verbose:
            rule("Evaluation")
        results = self._run(prompts)
        if self._verbose:
            rule("Evaluation complete", style="dim")
        return results

    def _validate_input(self, prompts: list[str]) -> None:
        if not prompts:
            raise ValueError("Evaluator.run() requires a non-empty prompt set.")

    def _run(self, prompts: list[str]) -> dict:
        """Evaluates `prompts` and returns the results dict (see module docstring)."""
        self._log(f"[1/5] Generating answers for {len(prompts)} eval prompts...")
        parent = self._parent_answers(prompts)
        engine = ChildInferenceEngine(
            self._config.adapter_dir,
            max_new_tokens=self._config.max_new_tokens,
            batch_size=self._config.inference_batch_size,
        )
        tuned = engine.generate_tuned(prompts)
        untuned = engine.generate_untuned(prompts)
        self._log("[1/5] Done: parent, tuned, and untuned answers ready.")

        judge = Judge(model_id=self._config.judge_model)
        self._log("[2/5] Judging tuned vs untuned quality (primary)...")
        tuned_vs_untuned = self._win_rate(judge, prompts, tuned, untuned)

        self._log("[3/5] Judging parent-style match (tuned vs untuned)...")
        # Distillation objective: is the tuned child more like the parent's style
        # than untuned? Only as good as the reference -- the true styled teacher
        # when parent_instruction is set, else the parent with no system prompt.
        parent_likeness = self._win_rate(
            judge, prompts, tuned, untuned,
            references=parent, rubric=STYLE_JUDGE_INSTRUCTION,
        )

        self._log("[4/5] Judging tuned vs parent (gap to teacher)...")
        tuned_vs_parent = self._win_rate(judge, prompts, tuned, parent)

        self._log("[5/5] Scoring embedding similarity to the parent...")
        similarity = self._similarity(tuned, untuned, parent)
        self._log("[5/5] Done.")

        return {
            "adapter_dir": self._config.adapter_dir,
            "parent_model": self._config.parent_model,
            "judge_model": self._config.judge_model,
            "embedding_model": self._config.embedding_model,
            "num_prompts": len(prompts),
            "swap_positions": self._config.swap_positions,
            "comparisons": {
                "tuned_vs_untuned": tuned_vs_untuned,
                "parent_likeness": parent_likeness,
                "tuned_vs_parent": tuned_vs_parent,
            },
            "similarity_to_parent": similarity,
            "samples": self._samples(prompts, parent, tuned, untuned),
        }

    def _parent_answers(self, prompts: list[str]) -> list[str]:
        """Teacher answers, in the same order as `prompts`. The parent generates
        with the training recipe (its own max_tokens/temperature), so the
        reference matches the actual teacher rather than the child's eval knobs."""
        model = Model(
            model_id=self._config.parent_model,
            temperature=self._config.parent_temperature,
            max_tokens=self._config.parent_max_tokens,
        )
        if self._config.parent_instruction:
            model.set_instruction(self._config.parent_instruction)

        from fastsft.data.response_generator import ResponseGenerator

        distiset = ResponseGenerator(model=model).generate(prompts)
        # Key by instruction (unique after dedup) to survive row reordering;
        # assert_generation rejects a None/empty completion before it corrupts a
        # downstream judge or embedding.
        by_prompt = {
            row["instruction"]: model.assert_generation(
                row["generation"], row["instruction"]
            )
            for row in distiset["default"]["train"]
        }
        return [by_prompt[prompt] for prompt in prompts]

    def _win_rate(
        self,
        judge: Judge,
        prompts: list[str],
        a_answers: list[str],
        b_answers: list[str],
        references: list[str] | None = None,
        rubric: str = COMPARISON_JUDGE_INSTRUCTION,
    ) -> dict:
        """Win rate of `a_answers` over `b_answers`, judged in both A/B orders
        (when swap_positions) and averaged per prompt to cancel position bias.

        With `references` supplied, each pair is judged by resemblance to its
        reference under `rubric` (parent-likeness) via `compare_to_reference`,
        instead of head-to-head quality via `compare_samples`.
        """
        ids = [str(i) for i in range(len(prompts))]

        def verdicts_for(first: list[str], second: list[str]) -> dict[str, Verdict]:
            """Judge one ordering: `first` in the 'A' slot, `second` in 'B'."""
            if references is None:
                quality_pairs = {
                    i: (prompts[int(i)], first[int(i)], second[int(i)]) for i in ids
                }
                return judge.compare_samples(quality_pairs, prompt=rubric)
            reference_pairs = {
                i: (prompts[int(i)], references[int(i)], first[int(i)], second[int(i)])
                for i in ids
            }
            return judge.compare_to_reference(reference_pairs, prompt=rubric)

        # Order 1: a is presented as "A". Order 2 (swap): a is presented as "B".
        verdicts1 = verdicts_for(a_answers, b_answers)
        verdicts2 = {}
        num_orders = 1
        if self._config.swap_positions:
            verdicts2 = verdicts_for(b_answers, a_answers)
            num_orders = 2

        per_prompt = []
        for i in ids:
            credit = self._credit(verdicts1[i].winner, a_label="A")
            if self._config.swap_positions:
                credit += self._credit(verdicts2[i].winner, a_label="B")
            per_prompt.append(credit / num_orders)

        wins = sum(1 for score in per_prompt if score > _TIE)
        ties = sum(1 for score in per_prompt if score == _TIE)
        losses = sum(1 for score in per_prompt if score < _TIE)
        return {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": sum(per_prompt) / len(per_prompt),
            "orders_judged": num_orders,
        }

    @staticmethod
    def _credit(winner: str, a_label: str) -> float:
        """Quality credited to the first answer (`a`), given which label it held."""
        if winner == "tie":
            return _TIE
        return _WIN if winner == a_label else _LOSS

    def _similarity(
        self, tuned: list[str], untuned: list[str], parent: list[str]
    ) -> dict:
        model_id = self._config.embedding_model
        tuned_sims = pairwise_similarities(tuned, parent, model_id)
        untuned_sims = pairwise_similarities(untuned, parent, model_id)
        return {
            "tuned_vs_parent": _mean(tuned_sims),
            "untuned_vs_parent": _mean(untuned_sims),
        }

    def _samples(
        self, prompts: list[str], parent: list[str], tuned: list[str], untuned: list[str]
    ) -> list[dict]:
        """A few worked examples for qualitative inspection in the viewer."""
        return [
            {
                "prompt": prompts[i],
                "parent": parent[i],
                "tuned": tuned[i],
                "untuned": untuned[i],
            }
            for i in range(min(_SAMPLES_KEPT, len(prompts)))
        ]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
