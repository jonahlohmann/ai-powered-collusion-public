# AI-use record

This file records how AI tools were used for the bachelor's thesis behind this repository. It
is condensed from the author's working record, which holds one row per session (about 160
rows). Here, sessions are merged into the working steps of the thesis, in the order of the
work, each with its main prompt. It details Section 4.8 of the thesis, which remains the
authoritative description, and supplements the declaration of aids.

## Tools and division of work

| Tool | Use |
|:--|:--|
| Claude Code (Anthropic) | Carrying out the prompts in the repository: simulation code, testing suite and analysis scripts of the simulation runs, including the runs' reports, figures and tables; LaTeX drafts from the author's outlines; grammar and sanity checks of the author's text; the interactive page of the results in this repository (`results_explorer.html`, step 19). Models: Claude Opus 4.7, Claude Fable 5.1 and 5.2, Claude Opus 5.5 and Claude Sonnet 5.5 |
| Claude chat (claude.ai Project, Anthropic) | Working through DGJ's study and the adjacent literature, brainstorming, explanations, feedback on drafts, translation, and optimising the prompts for Claude Code |
| Google Antigravity IDE | Development environment and editor |
| Zotero | Reference management; in coordination with it, Claude Code located the page numbers of statements made in the cited sources' abstracts, introductions and conclusions |

The workflow was a tandem between chat and coding instances. In the chat, the author worked
through DGJ's study and the adjacent literature and optimised the prompts, which were then
carried out by Claude Code in the repository. Based on the author's insights, Claude Code
helped write the simulation code, the testing suite and the analysis scripts of the
simulation runs. The output of both was steered through continuous management and updates
of system memory. Whenever a working step involved a choice, the decision stayed with the
author.

For the text, the author outlined the structure and the sources and drafted the content of
each chapter and section. Claude Code turned this into a LaTeX draft, which the author used
as inspiration and rewrote. A grammar and sanity check by Claude Code followed. AI was never
used as a source.

## How to read the quoted prompts

Prompts are reproduced as they were given, apart from corrected spelling. `[...]` marks an
omission. File names inside a quoted prompt refer to the author's private working
repository, and many of those files (plans, logs, state files, the thesis sources) are not
part of this repository. Run labels inside a quoted prompt are the working labels of the
time, and "the paper" there means Dou et al. (2025), called DGJ elsewhere in this file. The
table maps the working labels to the labels of the thesis.

| Working label in a prompt | Label in the thesis and in this repository |
|:--|:--|
| R2, M2, main run (`results/m2`, `config_m2.yaml`, `params_paper_nv5.yaml`) | R1, the baseline (`results/r1_baseline`) |
| Phase 0 (`results/production_phase0`, `config_phase0.yaml`) | R1.1 (`results/r1_1_early_stop`) |
| R1, M1 (`results/m1_long`, `params_thesis.yaml`) | R1.2 (`results/r1_2_earlier_spec`) |
| R5 (`results/m2_snapped`) | R1.3 (`results/r1_3_rounded_price`) |
| R10a | R1.4 (`results/r1_4_no_price`) |
| R10b | R1.5 (`results/r1_5_no_price_beta3e6`) |
| R9 (`results/m2_beta5e7`) | R1.6 (`results/r1_6_ref_beta`) |
| R8, paper_exact | R1.7 (`results/r1_7_ref_grid`) |
| m2_sigma01 | R1.8 (`results/r1_8_ref_noise`) |
| R3 (`results/qa`) | R2, Q–AC (`results/r2_q_ac`) |
| R4 (`results/aa`) | R2.1, AC–AC (`results/r2_1_ac_ac`) |
| R7, alpha_het | R2.2 (`results/r2_2_two_alphas`) |
| V1, V2 (`results/br_q`, `results/br_ac`) | validation runs (`results/validation_q`, `results/validation_ac`) |
| T1 (`results/qa_tune_a`, `_b`, `_c`) | pilot (`results/pilot_a`, `_b`, `_c`) |

Note that the working labels R1 and R2 do not mean the R1 and R2 of the thesis. In a quoted
prompt, "snapped" is the rounded price and "actor-critic" is the AC learner.

## Working steps

### 1. Project set-up and first code base

- **Tool and model:** Claude chat (model not recorded) for working through DGJ's study and
  for the prompts; Claude Code, Claude Opus 4.7, for the code.
- **Purpose:** Set up a replication of DGJ at laptop scale and build the simulation:
  benchmarks, grids, market environment, Q-learner, metrics, runner and tests; first
  parameter tuning and a first set of 30 sessions.
- **Main prompt:** Prompts not kept as files; given in the sessions. Summary: for each build
  step, a specification was worked out in the chat from DGJ's text (closed-form benchmarks,
  then discretisation, then environment, agent and runner) and given to Claude Code, which
  implemented it with tests.
- **What it produced:** The first versions of `src/ai_trading/simulation/benchmarks.py`,
  `env/discretization.py`, `env/market.py`, `agents/base.py`, `agents/q_learning.py`,
  `simulation/metrics.py`, `simulation/runner.py` and of `tests/`. The results of this step
  are superseded (step 2) and are not in this repository; an early plan to pair the
  Q-learner with PPO was dropped.
- **Where it entered the thesis:** The code base behind chapters 3 and 4. No result of this
  step is reported.

### 2. Reassessment of the code base

- **Tool and model:** Claude Code (model not recorded).
- **Purpose:** Check every module, script and test against DGJ's article and Online Appendix
  before the project was resumed.
- **Main prompt:** Prompt not kept as a file; given in the session. Summary: the author asked
  for a full reassessment of the repository against DGJ, line by line, with the claims
  verified on the saved data where possible.
- **What it produced:** A report of findings: profits were computed on the rounded price;
  the shocks of the impulse-response test were too small and had the wrong sign; the price
  grid did not follow DGJ's formula; one line of the market maker took most of the running
  time.
- **Where it entered the thesis:** Through the corrected pipeline of step 3; the rounded
  price as a robustness run (R1.3) in chapters 4 and 5.

### 3. Corrected pipeline

- **Tool and model:** Claude Code (model not recorded).
- **Purpose:** Correct the simulation and measurement pipeline after the reassessment,
  without changing the research design.
- **Main prompt:** An implementation plan, written by Claude Code from the report of step 2
  and then handed to a coding agent as its prompt. Its defining part:

  > **Executor:** a coding agent working in this repository (any model tier). Everything you
  > need is in this file. [...] A review found that the 30-session baseline result (Δ^C =
  > 0.68, git tag `baseline-v1`) is mostly an artefact. Phase 0 fixes the pipeline so the 30
  > sessions can be re-run on correct code. [...]
  >
  > A | Profits are computed on the **raw** clearing price `p = v̄ + λ̂·y`. The price grid only
  > indexes the state. A `reward_price: snapped` option is kept for A/B comparison. [...]

- **What it produced:** Rewards on the raw price, the price grid by DGJ's formula, the
  evaluation mode, independent random-number streams, per-path Δ^C, the counterfactual
  impulse-response test and the session flags of the runner in `src/ai_trading/`;
  `scripts/baseline_report.py`; the 30 sessions of R1.1
  (`results/r1_1_early_stop/report.txt`).
- **Where it entered the thesis:** Chapter 4 (Sections 4.1 to 4.6); R1.1 in the sensitivity
  table of chapter 5.

### 4. Execution plan and project set-up for the thesis

- **Tool and model:** Claude Code, Claude Fable 5.1 (plan); Claude Sonnet 5.5 and Claude
  Opus 5.5 (applying the author's decisions, thesis template, progress dashboard).
- **Purpose:** Analyse the 30 sessions of R1.1 and turn the remaining work into a sequenced
  plan with one ready prompt per work package.
- **Main prompt:**

  > # Task: Produce the final execution plan for the remaining thesis work
  >
  > You are the planning lead for a bachelor thesis repo. Your output is a PLAN, not code.
  > A less capable model (Sonnet 5.5 / Opus 5.5) will execute it step by step, so it must be
  > concrete, sequenced and unambiguous. [...]
  > - [...] Prefer simple, debuggable changes. Every code step must be verifiable by tests
  >   or a clear numeric check.
  > - **Primary goal: finish the thesis with valuable, defensible takeaways.** Moving closer
  >   to the paper is secondary. [...]
  >
  > The **30-session Phase 0 re-run is complete** at `results/production_phase0/` [...]. Before
  > making any decision, analyse it read-only with the existing report scripts [...]. These
  > results are the empirical basis for every decision below. [...]

- **What it produced:** A plan with decisions for the author to take (stopping rule,
  parameters, the second learner type), a run register and the prompts of steps 5, 7, 9, 10
  and 11. Two observations from the analysis: sessions ended at the cap of periods before
  they converged, and the action grid's own Nash equilibrium is not at Δ^C = 0. Also a LaTeX
  template for the thesis and a progress dashboard, neither of which is in this repository.
- **Where it entered the thesis:** No text. The decisions the author took from it shape
  Sections 4.3 to 4.5.

### 5. Training diagnostics, stopping rule and run R1.2

- **Tool and model:** Claude Code, Claude Opus 5.5.
- **Purpose:** Measure whether sessions have settled, compute the Nash equilibrium of the
  action grid, and rerun the earlier specification with a longer window and a higher cap.
- **Main prompt:**

  > [...] Execute work package WP1 exactly as specified here. Do not add features that are
  > not listed.
  >
  > RULES
  > - Existing configs must produce identical results. All new behaviour is behind new config
  >   keys whose defaults reproduce today's behaviour. [...]
  >
  > STEP 2: grid stage game in simulation/benchmarks.py [...]
  > STEP 3: CLI flag [...] add `--max-periods N` [...]
  > STEP 4: training diagnostics in runner.run_session [...]

- **What it produced:** `grid_stage_game` in `src/ai_trading/simulation/benchmarks.py`; the
  stability diagnostic, the "settled" flag and `trajectory.parquet` in
  `simulation/runner.py`; the report on all, converged and settled sessions in
  `scripts/baseline_report.py`; `experiments/01_replicate_baseline/config_r1_2_earlier_spec.yaml`
  and `results/r1_2_earlier_spec/report.txt`.
- **Where it entered the thesis:** Section 4.5 (convergence), the grid Nash value that is
  reported next to Δ^C, the learning curves (Figure 4), R1.2 in chapter 5.

### 6. Deviation test and staleness check

- **Tool and model:** Claude Code, Claude Opus 5.5.
- **Purpose:** Test whether a single best response pays under the learned policies, and
  whether the learners' stored values for that deviation are up to date.
- **Main prompt:** Prompt not kept as a file; given in the session. Summary: after the report
  on R1.2, the author asked whether the learned outcome is a Nash equilibrium; Claude Code
  suggested a design for the test, the author decided on it and asked for it as a standard
  script with tests.
- **What it produced:** `scripts/deviation_test.py`, `tests/test_deviation_test.py`, and
  `results/<run>/deviation_test.txt` for every run.
- **Where it entered the thesis:** Section 4.6 (method), Section 5.2, Appendix A.3 and
  Table 8.

### 7. The baseline specification

- **Tool and model:** Claude Code, Claude Fable 5.1 (implementation, with a code review by a
  second agent of the same model); Claude Opus 5.5 (applying the author's five decisions).
- **Purpose:** Bring three choices in line with DGJ's description: an action grid per value,
  the exploration counter per value, and DGJ's initialisation of the Q-values.
- **Main prompt:**

  > [...] Execute work package WP3 exactly as specified here.
  >
  > RULES
  > - Existing configs must produce identical results. The three new keys default to today's
  >   behaviour: action_grid "global", exploration_counter "state", q_init "optimistic". [...]
  >
  > STEP 1: action table (env/discretization.py, env/market.py) [...]
  > - New `build_action_table(mode, n_x, v_grid, ..., v_bar)` returning shape (n_v, n_x):
  >   mode "global": every row is build_action_grid(...) (today's 1-D grid);
  >   mode "per_v": row k = chi_grid * (v_grid[k] - v_bar).
  >   In per_v mode action index a means aggressiveness level a at every v [...]

- **What it produced:** The config keys `action_grid`, `exploration_counter` and `q_init`
  in `src/ai_trading/`; `experiments/shared/params_baseline.yaml` and
  `experiments/01_replicate_baseline/config_r1_baseline.yaml`; the handling of degenerate
  states; the separate lines for visited states in the impulse-response part of the report.
- **Where it entered the thesis:** Sections 4.2 to 4.5 and Table 5 (parameters next to
  DGJ's values).

### 8. The baseline run R1

- **Tool and model:** Claude Code, Claude Sonnet 5.5 (launch) and Claude Fable 5.1
  (completion and reports). The prompt was optimised in the Claude chat with Claude Opus 5.5.
- **Purpose:** Run the 30 sessions of the baseline unattended and write its reports.
- **Main prompt:**

  > [...] Your task: launch R2a and R2b so that they run unattended with the laptop lid
  > closed [...], and write the reports when the researcher is back. You change no code and
  > no config.
  >
  > RULES
  > - No edits under src/, tests/, experiments/. Never stage anything under results/. [...]
  > - Do not decide Gate G1. Compute its numbers; the researcher decides [...].
  > [...]

- **What it produced:** `results/r1_baseline/` with its logs (`logs/s*.log`), `report.txt`
  and `deviation_test.txt`. The shell scripts that started the sessions on the laptop are
  not part of this repository.
- **Where it entered the thesis:** Chapter 5 (all numbers of the baseline). The reports
  state numbers without interpretation.

### 9. AC learner, validation runs, pilot and the heterogeneity runs

- **Tool and model:** Claude Code, Claude Fable 5.1.
- **Purpose:** Add a tabular softmax AC learner and a rival with a fixed trading
  intensity; validate both learners against that rival; pick the AC learner's settings in a
  pilot by a rule fixed in advance; run Q–AC (R2) and AC–AC (R2.1).
- **Main prompt:**

  > [...] Execute work package WP4 exactly as specified here. [...]
  > - Do not tune hyperparameters beyond the defaults and the three pilot settings below.
  >
  > STEP 1: agents/actor_critic.py, class ActorCriticAgent(Agent), registry key "actor_critic"
  > [...] own keys with defaults: alpha_actor 0.1, alpha_critic 0.05, entropy_coef 0.1,
  > entropy_decay (default: the value of beta), reward_scale (default: pi_M). [...]
  >   delta = r + rho * V[s2] - V[s]
  >   V[s] += alpha_critic * delta
  >   [...]
  >       H[s][b] += alpha_actor * (d * grad_logp + eta * grad_ent)

- **What it produced:** `src/ai_trading/agents/actor_critic.py`, `agents/fixed_policy.py`
  and their tests; the configs under `experiments/02_heterogeneity/`; `results/validation_q`,
  `results/validation_ac`, `results/pilot_a`, `_b`, `_c`, `results/r2_q_ac` and
  `results/r2_1_ac_ac` with reports and deviation tests.
- **Where it entered the thesis:** Section 4.3 (the AC learner; the description is the
  author's), chapter 6, Appendix A.2 and Tables 6 and 7.

### 10. Robustness and reference runs, trigger test and replay of the impulse responses

- **Tool and model:** Claude Code, Claude Fable 5.1.
- **Purpose:** Prepare and run the configurations that change one setting each (rounded
  price, two learning rates, DGJ's value grid and exploration decay, a slower decay), and add
  two descriptive scripts on the learned policies.
- **Main prompt:**

  > [...] Execute work package WP6. No changes under src/. [...]
  >
  > PART A: configs (each a copy of the main config with only the listed changes and a header
  > comment saying what differs and why)
  > 1. [...] config_m2_snapped.yaml: overrides.reward_price "snapped" [...]. Purpose: A/B for
  >    the snapped-price artefact on current code.
  > 2. [...] config_alpha_het.yaml: agents [{type: qlearner}, {type: qlearner, config:
  >    {alpha: 0.1}}] [...]. Purpose: the paper's own heterogeneity (Online Appendix 4.12).
  > 3. [...] params_paper_exact.yaml: [...] n_v: 10, beta: 5.0e-7, max_periods: 400000000
  > 4. [...] config_m2_beta5e7.yaml: overrides.beta 5.0e-7 [...]
  > PART B: scripts/qtable_trigger_test.py [...] Direct test for price-conditional punishment
  > in the saved greedy policies, without shocks. [...]

- **What it produced:** `config_r1_3_rounded_price.yaml`, `config_r1_6_ref_beta.yaml`,
  `config_r1_7_ref_grid.yaml` with `experiments/shared/params_ref_grid.yaml`, and
  `experiments/02_heterogeneity/config_r2_2_two_alphas.yaml`; the four result folders with
  reports and deviation tests; `scripts/qtable_trigger_test.py` and `scripts/irf_replay.py`
  with tests, and their output in `results/r1_baseline/`.
- **Where it entered the thesis:** Table 3 (sensitivity) and Section 5.3; R2.2 in chapter 6.
  The two scripts are descriptive; their output is not quoted in the thesis.

### 11. Figures, tables and number macros

- **Tool and model:** Claude Code, Claude Fable 5.1 (first version), Claude Sonnet 5.5 and
  Claude Opus 5.5 (parameter table), Claude Fable 5.2 (all later changes).
- **Purpose:** One script that turns the saved results into every figure, table and quoted
  number of the thesis, so that no number is typed by hand.
- **Main prompt:**

  > [...] Execute work package WP5: implement the thesis figures and tables. You write no
  > interpretation and no thesis prose.
  >
  > RULES
  > - No changes to simulation behaviour. Only src/ai_trading/viz/figures.py, a new script
  >   scripts/make_thesis_outputs.py, and tests for pure helper functions. [...]
  >
  > SCRIPT
  > scripts/make_thesis_outputs.py [--runs KEY=DIR ...] [--out-figures thesis/figures]
  > [--out-tables thesis/tables]. [...] A missing run produces a warning and its outputs are
  > skipped. [...]

  A second prompt entered the reasons of the parameter table: "Task: fill the 'Reason'
  column of table T2 [...] and add the missing design rows. Small, self-contained change to
  the table generator only." The reason texts in that prompt were the author's. Later changes
  (the overview of runs, the drawing of a learner's table, the paths of the two learners in
  Q–AC, the split parameter tables, labels, heads and notes) were made on the author's
  feedback lists in the sessions; those prompts were not kept as files.
- **What it produced:** `src/ai_trading/viz/figures.py`, `scripts/make_thesis_outputs.py`
  and their tests; everything under `outputs/figures` and `outputs/tables`, including
  `outputs/tables/numbers.tex`.
- **Where it entered the thesis:** Every figure and table and every number quoted in the
  text. The figure notes name Claude Code as the tool that drew them. Notes and captions
  generated by the script were proposed by Claude Code and reworded by the author.

### 12. Runs without the lagged price in the state

- **Tool and model:** Claude chat, Claude Fable 5.2 (prompt, with the reading of each
  possible outcome fixed before the run); Claude Code, Claude Opus 5.5 (execution).
- **Purpose:** DGJ's own check of the price-trigger mechanism: remove the lagged price from
  the learners' state and see what remains (R1.4 and R1.5).
- **Main prompt:**

  > [...] Your task: add one default-off switch that removes the lagged price from the
  > agents' state, prove that the default behaviour is unchanged, run R10a and R10b in the
  > background, report them, regenerate the thesis outputs and update the docs. The
  > researcher authorised this exception to the code freeze [...].
  >
  > RULES
  > - The ONLY change under src/ is the state_price switch of STEP 2. [...]
  > - No interpretation of Δ^C in logs. Print numbers and the comparison defined in STEP 6.
  > [...]
  > - "none": the first component of the state returned by reset() and step() is always 0.
  >   Nothing else changes [...]

- **What it produced:** The key `state_price` in `src/ai_trading/env/market.py` with eight
  tests; `config_r1_4_no_price.yaml` and `config_r1_5_no_price_beta3e6.yaml`;
  `results/r1_4_no_price` and `results/r1_5_no_price_beta3e6` with reports; the macros for
  differences between runs in `scripts/make_thesis_outputs.py`.
- **Where it entered the thesis:** Section 5.2 and Table 3. The author decided on the
  reading of the result.

### 13. Reference run at DGJ's noise level

- **Tool and model:** Claude Code, Claude Fable 5.2.
- **Purpose:** Rerun the baseline specification with DGJ's noise level and nothing else
  changed (R1.8).
- **Main prompt:** Prompt not kept as a file; given in the session. Summary: while working on
  chapter 4, the author asked for a copy of the baseline configuration with one change (the
  noise level), launch scripts for it and, once it had finished, its reports and the
  regenerated outputs.
- **What it produced:** `experiments/01_replicate_baseline/config_r1_8_ref_noise.yaml`,
  `results/r1_8_ref_noise` with report and deviation test, the rows of R1.8 in the generated
  tables.
- **Where it entered the thesis:** Section 4.4, Table 3 and Section 5.3; the first limit of
  Section 7.2.

### 14. Exploratory run on the price grid

- **Tool and model:** Claude Code, Claude Fable 5.2.
- **Purpose:** Try a price grid built for each value, which has the property that Dou et al.
  (2025, p. 23, footnote 25) describe, outside the 12 runs of the thesis.
- **Main prompt:** Prompt not kept as a file; given in the session. Summary: the author asked
  for a switch for such a price grid in a copy of the repository, so that the frozen code
  stayed unchanged, for the configuration of R1.8 to be run with it, and for one session
  with a cap of one billion periods.
- **What it produced:** `experiments/03_exploratory_price_grid/` (patch, tests, two
  configurations, reports and a README).
- **Where it entered the thesis:** A short passage in the first limit of Section 7.2, drafted
  by Claude Code on the author's decision and rewritten by the author with the chapter
  (step 15).

### 15. LaTeX drafts of the chapters from the author's outlines

- **Tool and model:** Claude Code, Claude Fable 5.2 and Claude Opus 5.5.
- **Purpose:** Turn the author's outline of a chapter (structure, sources, content, often
  dictated) into a LaTeX draft that the author then rewrites.
- **Main prompt:** The author's first instruction of this kind (voice input):

  > I want you to structure and reformat the things that I'm saying freely here into
  > a LaTeX format, enriched with the knowledge I have gathered in this project and the
  > sources I am providing you with.
  >
  > Basically, what I'm going to do is I am going to broadly speak about the things that I
  > would like to see in a certain chapter and the structure in which I want to say it [...].
  > Use the sources I quote to construct my words into example passages, that I then rewrite.
  > I will restructure sentences that I don't like. I will add things, subtract things. And the
  > core of the arguments made remains my doing. [...]

- **What it produced:** LaTeX drafts for chapters 1 to 8, kept apart from the chapter files,
  that the author used as inspiration. None of these drafts are in this repository.
- **Where it entered the thesis:** The author rewrote these drafts into chapters 1 to 8 and
  the abstract as they stand in the thesis.

### 16. Grammar and sanity checks of the author's text

- **Tool and model:** Claude Code, Claude Fable 5.2 and Claude Opus 5.5.
- **Purpose:** Check each chapter after the author had written it: grammar, numbers against
  the run reports, consistency across chapters, and, in coordination with Zotero, the page
  numbers of statements made in the cited sources' abstracts, introductions and conclusions.
- **Main prompt:** Prompts not kept as files; given in the sessions. The requests as
  recorded: a "complete correctness check, content and grammar" per chapter; for the figures
  and tables and for the final pass, grammar only. The author answered each list point by
  point and worked the points into the thesis.
- **What it produced:** Numbered lists of problems, each with location, reason and a
  proposed replacement; a check of the whole thesis (build, numbers, cross-references, page
  numbers of the citations); a final grammar check.
- **Where it entered the thesis:** Chapters 1 to 8. The author revised and finalised the
  chapters. Titles, captions and layout settings were written by the author.

### 17. Appendix, declaration of aids and front matter

- **Tool and model:** Claude Code, Claude Fable 5.2 and Claude Opus 5.5.
- **Purpose:** Draft and check the appendix in the same way as the chapters (steps 15 and
  16). Keep the declaration of aids and the lists in line with the thesis.
- **Main prompt:** Prompt not kept as a file; given in the session. Summary: the author gave a
  list with general rules and points per section (wording, a split of the parameter
  table, the reason for the tie rule of the pilot, a larger deviation table, the placement of
  Figure 9).
- **What it produced:** LaTeX drafts for the appendix; Tables 5 to 8 and Figures 8 and 9
  through `scripts/make_thesis_outputs.py`; rows of the declaration of aids on the author's
  account of the tools used.
- **Where it entered the thesis:** The appendix, the declaration of aids.

### 18. This public repository

- **Tool and model:** Claude chat, Claude Fable 5.2 (a first prompt, not carried out in that
  form); Claude Code, Claude Fable 5.2 (this copy).
- **Purpose:** Publish the code, the configurations, the results and this record in a
  repository that stands on its own.
- **Main prompt:** The author's instruction:

  > Make a fully local copy of the repo, to be connected to GitHub, to create the perfectly
  > slimmed down version without the clutter of my research [...]. The repo should be able to
  > completely stand on its own feet. This is so our work is completely replicable in a
  > separated environment which is a hard requirement for this thesis
  > - General: We only require things that went directly into our thesis, remove the clutter
  > [...]
  > - Yes, only the well-documented code which was used to run our simulations or products
  >   thereof
  > - Yes, run logs and reports of our cited simulations
  > - Yes, ai-use-record
  > - YES, map of implementation choices

- **What it produced:** This repository: the code with comments and docstrings set to the
  labels of the thesis and no statement changed, the configurations, the results of the 12
  runs, the validation runs and the pilot, the generated figures and tables, the
  documentation under `docs/` and the README files, prepared by Claude Code on the author's
  instructions.
- **Where it entered the thesis:** Section 4.8 points to this repository for further records
  of the AI use.

### 19. Interactive page of the results

- **Tool and model:** Claude Code, Claude Fable 5.2.
- **Purpose:** Let a reader of this repository explore the results of the runs in a browser,
  without installing anything.
- **Main prompt:** The author's instruction:

  > Create an HTML file to dynamically interact with the findings of our runs. Come up with
  > something nice and playful but informative. The HTML file should be able to open for
  > anyone who uses this public repo [...]. Put the file in the repo, name it accordingly,
  > and point to it in the README.

- **What it produced:** `results_explorer.html`, a single page that contains its data: the
  one-period game on the action grid to play, the learning paths of the runs, the sessions
  of every run on the scale of Δ^C, the responses to a forced deviation, the deviation test
  per session and a table of every session. `scripts/make_results_explorer.py`, which writes
  the data of the page from `results/` with the functions of
  `scripts/make_thesis_outputs.py`, so that no result is typed into the page by hand. The
  section "Explore the results" of the README and a row in `scripts/README.md`. The layout,
  the charts and the explanatory texts of the page were written by Claude Code; the texts
  are condensed from the README and `docs/implementation-choices.md`.
- **Where it entered the thesis:** Nowhere. The page shows the results that the thesis
  reports; no figure, table or sentence of the thesis draws on it.
