# rules.md - Rules for every human and every AI assistant on this repo

Read PRD.md and this file completely at the start of every session, before writing any code. If a request conflicts with them, stop and tell the user instead of silently following either.

## 1. Follow
1. PRD.md is the source of truth. Build only what it lists, in tier order: Must, then Should, then Could.
2. Ask before assuming. If something is unclear or missing, say what is unclear and ask. Do not fill the gap with a guess.
3. Say "I don't know" or "I could not verify this" when that is the truth.
4. Verify before using any model ID, library, function, dataset or API. Run scripts/check_models.py for models and check the library's installed version for APIs. Never rely on memory for names or arguments.
5. Keep module contracts stable: function signatures and the column names of dataset.csv and features.csv are shared agreements. Changing one requires a docs/DECISIONS.md entry and telling the whole team in the same commit message.
6. Split data by image (70/15/15) with a fixed random seed of 42. The test split is used only for final evaluation. Tune only on train and validation.
7. Use the same test split for every baseline, ablation and the final model.
8. Cache every external API response (for example Gemini) to disk. Re-runs must not call the API again for the same input.
9. Keep secrets in .env only (never in code, notebooks, logs or commits). Keep .env.example up to date with variable names only.
10. Pin every dependency in requirements.txt and add it in the same commit that first uses it.
11. Write small functions with type hints and docstrings. Prefer clear names over comments. Log with the logging module, not scattered prints.
12. Work on your own branch, make small commits with clear messages, and pull before you push. Read a file before editing it. Edit only the modules you own unless the owner agreed.
13. Record every real result (metric, plot, table) by running code that produces it, and save the generated output file with its script name and date.
14. Keep the demo honest: show the claim, the evidence (BLIP caption), the verdict and the confidence. Show the sentence "Consistency is not proof of truth."
15. State limitations plainly in the README: synthetic data, label noise rate from the 100-sample hand-check, caption hallucination, and bias in pretrained models.
16. Disclose AI assistance in the README, as the hackathon may require.

## 2. Never
1. Never invent or estimate numbers, metrics, results, citations, dataset sizes, model names or links. If a number was not produced by running code, do not write it anywhere.
2. Never hard-code, mock or fake predictions, scores or results, even temporarily in the demo. Unfinished parts must raise NotImplementedError or show "not implemented".
3. Never fine-tune CLIP, BLIP or any other large model. Never add features outside the PRD without the team lead's written approval in docs/DECISIONS.md.
4. Never put the same image in more than one split. Never tune hyperparameters or choose thresholds on the test split. Never leak labels into features (for example by using the label-generation prompt or the label text as a feature).
5. Never change the label definitions, split ratios, metrics or evaluation protocol silently.
6. Never delete, rename or overwrite files or code written by someone else without asking. Never force-push, rewrite history or merge your own risky changes without review.
7. Never commit API keys, .env, large model weights, raw datasets or generated images. Respect .gitignore.
8. Never add a dependency, model or dataset without checking its licence and listing it in requirements.txt or the README.
9. Never use private, personal or non-permitted data or photos. Never present results as fact-checking or proof of truth.
10. Never claim that something works, passes or is finished without running it and reading the output. Report failures honestly.
11. Never use absolute paths from one machine, hidden global state, or unseeded randomness.
12. Never expand scope in the last 4 hours before submission: only bug fixes, README, demo video and submission.

## 3. When unsure
Stop, state the problem in one or two sentences, list the options, and ask the user. A short question is always better than a wrong guess that breaks a teammate's work.
