---
name: write-user-guide
description: Write or revise a user-facing guide under docs/guides/ in the Deep Learning Toolkit repository, covering the house structure, voice, and accuracy rules. Use this skill whenever the user asks for a guide, tutorial, walkthrough, how-to, usage doc, or "getting started" document, whenever they ask to document a toolkit feature, training workflow, or API for other people to follow, and whenever they ask to revise or extend an existing file under docs/guides/, even if they never say the word "guide".
---

# Writing a Deep Learning Toolkit (`dlk`) User Guide

A guide teaches a reader to do one thing, end to end, and leaves them understanding why it worked. Reference documentation lists what exists; a guide walks a path through it. Keep that distinction in mind, because it decides what belongs in the file and what does not.

## The reader

Assume a competent scientist or engineer who knows PyTorch but has never used this feature. They read fast, they like learning, and their time is expensive. Two consequences shape every sentence:

- **They can follow an explanation, so give them one.** Never reduce a guide to a command list. The commands are the cheap part; the reasoning is why anyone reads prose instead of a docstring.
- **They notice padding.** Cut sentences that restate the previous sentence, announce what you are about to say, or hedge. If a paragraph, sentence, or sentence part could be deleted without loss, delete it.

## Ground every claim in the source before writing

This is the rule that separates a guide people trust from one they stop reading. Documentation that describes intended behavior instead of actual behavior costs a reader an afternoon and costs the project their confidence.

Before writing a sentence about how something behaves:

1. **Read the implementation.** Library code lives under `dlk/`: training loops and optimization in `dlk/opt/`, network building blocks in `dlk/nets/`, losses in `dlk/loss/`, logging and run management in `dlk/mgmt/`. Trace an argument from the public function to the code that consumes it. The tests under `tests/` show the intended call patterns.
2. **Copy real strings.** Log lines, assertion messages, error text, and file names written by the code get quoted from the source, never invented. A reader searching for the message they actually saw has to find it in your guide.
3. **Run the commands.** Where it is safe and read-only, execute the commands you intend to publish and confirm the output matches what you claim. Use `uv run python ...` and the make targets (`make test`, `make lint`); the bare `python` shim is broken in this environment.
4. **Say so when you are unsure.** A hedge the reader can act on ("check your site's default") beats a confident sentence that is wrong at half of sites. Flag the uncertainty to the user too, so they can confirm or correct it.

The payoff shows up in the finished guide as specificity. Explaining that `create_distributed_sampler` must receive the base seed while every other random generator gets a rank-offset seed, because all ranks must agree on one shuffle permutation before slicing their shards from it, only becomes possible after reading `dlk/opt/distributed.py`. That single paragraph is worth more than a page of general advice.

## Structure

Guides follow a consistent shape. See `template.md` in this skill directory for a fill-in skeleton.

**Frontmatter.** `Title`, `Author`, `Co-Authored-By` when an agent helped, `Date`, and `tags`.

**Title.** A gerund phrase naming the task and the subject: "Training with DDP across Multiple GPUs". Mirror it in the `Title` frontmatter field.

**Opening, two or three short paragraphs.** State what the feature is, using one bolded term you define on the spot. Say why the obvious approach is awkward. Say what this guide does instead. Start with the subject matter; skip meta-preamble like "In this guide we will explore".

**A constraint admonition, when one exists.** Hard requirements such as a minimum torch version or a required launcher belong at the top, in a note admonition where nobody can miss them.

**A section of numbered steps** for the path every reader must walk. Head them `### Step N: <verb phrase>` and group them under `##` headings by phase ("Preparing the application", then "Launching on the cluster"). Numbering across the group rather than restarting per phase lets the text refer to "the five steps above".

**Optional sections after the required path**, separated by `---`. Common extensions that most readers eventually need go here. Open by saying the section is optional and naming the trigger for reading it. Inside, use topical `###` headings without numbers, since these are not sequential.

**A closing reference section**, conventionally "Things worth knowing". Bold-lead paragraphs of the form `**Topic.** Explanation.` for facts that matter but interrupt the flow: portability limits, HPC differences, what to re-run after changing your mind. Do not summarize the guide here; the reader just read it.

## Voice

**Define terms inline, at first use, in bold or italic.** One clause is usually enough: "the *world size*, the total number of processes in the run". The reader never leaves the page to understand a word.

**Address the reader directly and use imperatives for actions.** "Wrap the model after moving it to the device." "Pick a free port and export it." Reserve "we" for nothing.

**Follow every instruction with its reason.** The pattern that works: state the action, then the consequence of getting it wrong. "`shuffle=False` in the DataLoader matters. Shuffling now belongs to the sampler, and PyTorch rejects the combination with `ValueError: sampler option is mutually exclusive with shuffle`."

**Write sentences that move forward.** Avoid the "not X, but Y" construction and its relatives. Presenting a wrong answer before the right one makes the reader hold two things in mind to receive one. Say the true thing directly: write "`validation_fn` receives the unwrapped module" rather than "`validation_fn` does not receive the DDP-wrapped model, but instead the underlying module".

**Avoid n-dashes / m-dashes with rare exceptions.** These dashes break the flow and burden with another thought to keep in mind: "A leads to B—with X, Y, and Z—and results in C." This makes the reader forget what the sentence started with or what it wants to focus on. When m-dashes are used, do not put spaces around them; spaces are for n-dashes.

**Let one aside per section reward attention.** A short evaluative remark keeps a technical document alive: "which is a miserable bug to chase", "That reduction is worth appreciating." Use these sparingly and only where you mean them.

**Prefer the concrete noun.** "The sampler", "the process group", "the checkpoint" beat "the artifact" or "the component".

## Command blocks

Every code block sits between two sentences: one saying what the command does, one saying what to expect back. A block with no lead-in makes the reader reverse-engineer your intent.

- Tag shell blocks ` ```sh `, Python ` ```python `, and program output ` ```text `.
- Show real output, trimmed, with machine-specific parts replaced by placeholders: `/home/you/...`, `<node>`, `<port>`.
- Never publish a path from the author's machine. Guides are read by people who do not share your home directory or your username.
- Cover CPU-only and CUDA systems where they differ (the gloo against the nccl backend, process counts, `map_location`).
- Link to upstream documentation for anything whose interface changes on someone else's schedule, such as torchrun options or Slurm directives. Duplicating it guarantees the guide goes stale.
- Give copy-pasteable one-liners that derive values instead of hardcoding them: `--nproc-per-node=$SLURM_GPUS_PER_NODE` cannot drift out of sync the way a literal `4` can.

## Formatting conventions

- The docs are plain Markdown, read in editors and on GitHub; a migration to Zensical is planned. Write admonitions in the Zensical form, `!!! note "An optional title"` / `!!! tip`, where an empty line separates a 4-spaces indented message text; they read fine as plain text today and render after the migration.
- Do *not* wrap prose; write paragraphs, list items, etc. in one line; don't use line breaks.
- Separate top-level sections with `---` when the guide has distinct phases.
- Use lists over tables.
- Reference repository files by path in backticks, so readers can search for them: `dlk/opt/distributed.py`, `tests/opt/test_train_distributed.py`.

## Anti-patterns

Recognizing these in a draft is faster than avoiding them while writing, so reread with this list in hand:

- **Meta-preamble.** "This guide will walk you through..." Delete and start with the subject.
- **Undefined jargon.** Any term a newcomer would have to look up needs a clause of definition at first use.
- **Commands without reasons.** A step that says what to type and not why is a step the reader cannot adapt when their situation differs.
- **"Not X, but Y."** Also "rather than X, Y" used purely for contrast. State the fact.
- **n-dashes / m-dashes.** Are often used for contrast that's distracting; use very sparingly and with good reason.
- **Restating the code.** Explaining that `wrap_ddp` calls `DistributedDataParallel` helps nobody. Explaining that it returns the model unchanged in a single-process run, so applications wrap unconditionally, helps everybody.
- **A summary section.** The reader finished the guide thirty seconds ago.
- **Author-machine paths, hardcoded versions in prose, and unverified output.**

## Working with the user

Draft first, then iterate. Produce a complete draft rather than an outline, since reacting to real prose is faster than imagining it.

After delivering a draft, tell the user three things: what you verified against the source, what you assumed, and what you are unsure about. The last one matters most, since it is where the user's domain knowledge is worth the most and your guessing is worth the least.

Expect the user to edit the file directly. Preserve their wording on the next pass; if their edit introduces a typo, point it out instead of silently rewriting it.

## Reference material in this skill

- `template.md`: a skeleton with the section shape and frontmatter.
- `references/examples.md`: weak and strong versions of the same passage, grounded in `docs/features/2026.02__ddp__2-usage.md` and `dlk/opt/distributed.py`. Read this when the voice guidance above feels abstract.
