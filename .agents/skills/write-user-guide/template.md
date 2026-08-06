---
Title: <Gerund phrase naming the task and the subject>
Author: <name>
Co-Authored-By: <agent model, when one helped>
Date: <YYYY-MM-DD>
tags:
  - <topic>
  - <topic>
---

# <Gerund phrase naming the task and the subject>

<Paragraph 1: what the feature is. Introduce and define one bolded term on the spot. State what the task requires that a reader would not guess.>

<Paragraph 2: why the obvious approach is awkward, and what this guide does instead. End on the promise the guide keeps.>

!!! note

    <Hard constraint the reader must satisfy: a minimum torch version, a required launcher, a platform limitation. Say how the tooling reports a violation.>

## <Phase one, as a gerund phrase>

### Step 1: <Verb phrase>

<Sentence saying what the command does and why.>

```sh
<command>
```

<Sentence saying what to expect back, and what a failure looks like.>

### Step 2: <Verb phrase>

<Repeat the lead-in / block / expectation pattern.>

!!! tip

    <Optional refinement that improves the result without being required.>

### Step 3: <Verb phrase>

!!! note

    <When a step is explanatory rather than required, say so here so a reader in a hurry can skip it.>

## <Phase two, as a gerund phrase>

### Step 4: <Verb phrase>

<Introduce the toolkit-specific function or argument. Explain the design decision behind it when it differs from a neighboring API, since that is where readers form wrong assumptions.>

```python
<call form 1>    # <when to use this one; mark the recommended one>
<call form 2>    # <when to use this one>
```

### Step 5: <Verb phrase, usually "Read what X reports">

<Show the tool's real output so the reader can match what they see.>

```text
<real output, trimmed, with machine-specific parts replaced by placeholders>
```

<Explain what the output proves. Where the code tries several strategies, list them in the order it tries them.>

---

## <Optional extension, as a gerund phrase>

<Open by saying this section is optional and naming the trigger for reading it. Then define the new term the section introduces.>

### Why this cannot break <the thing the reader fears breaking>

<Address the fear directly and early. State the one rule that keeps the guarantee true, in bold.>

### How <the system> finds <the thing>

<Explain the mechanism, grounded in the source. This is usually the most valuable subsection in the guide, because it is the part a reader cannot infer from the API alone.>

### <Topical subsection>

<Topical `###` headings here, without step numbers, since these are not sequential.>

### A more reproducible alternative: <name>

<Where a second workflow exists, present it after the familiar one and say plainly what it buys and what it costs.>

---

## Things worth knowing

**<Topic>.** <Fact that matters but would interrupt the main flow.>

**<Topic>.** <Portability limit, HPC difference, or cleanup step.>

**Changing your mind.** <What to re-run after changing a choice made earlier, and how to turn the feature back off.>
