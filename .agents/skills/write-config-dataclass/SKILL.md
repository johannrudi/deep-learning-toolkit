---
name: write-config-dataclass
description: Write a typed config dataclass that holds the arguments for one torch constructor, resolving machine-dependent ones automatically. Use when the user asks to wrap a torch class or function in a config, to stop repeating the same argument set across scripts, or to pick parameters like num_workers or pin_memory automatically.
---

# Config dataclasses for torch constructors

A config dataclass is a typed argument set for one torch constructor. It records the choices the project always makes, resolves the machine-dependent ones from the machine it runs on, and hands the result to torch as keyword arguments. A script then carries one object instead of a dozen loose arguments.

The worked example is `DataLoaderConfig` in `dlk/data/loader.py`, tested in `tests/data/test_loader.py`. Read both before writing a new one.

## Target model tier

> **Medium model.** Claude Sonnet / GPT mid class / GLM

The shape below is fixed, and the script in step 11 checks that it holds. What needs judgement is narrower: which parameters the project decides every time, which the machine should decide, and what each heuristic reads.

## When to write one

Write one when the target is a torch class or function with many keyword parameters, and either several of them depend on the hardware or on each other, or the same argument set is assembled in more than one script.

Do not write one for a constructor with two or three parameters. Do not write one for an abstract base either: `torch.optim.Optimizer` takes `(params, defaults)` and `torch.optim.lr_scheduler.LRScheduler` takes `(optimizer, last_epoch)`, so the target is always the concrete subclass, `AdamW` rather than `Optimizer`. A family of subclasses gets one config each, as `AdamConfig`, `AdamWConfig`, and `SGDConfig` do.

## Two shapes

Most configs take a **torch constructor** as their target, and `DataLoaderConfig` is the example.

Where the configuration lives in a composition rather than in one constructor, the target is instead the **`dlk` function that composes it**. `LinearConstCosineConfig` configures `create_linear_const_cosine_scheduler`, because the three torch schedulers it builds take 3 and 3 and 4 parameters, none of which is what a script wants to set, and the values it does want (`linear_epochs`, `init_learning_rate`) are transformed before any torch constructor sees them.

Everything below applies to both. Wherever it says parameter of the target, read parameter of that function, and the drift test in step 10 inspects that function's signature. This is the only case where wrapping `dlk`'s own code is right, and it is worth checking that a composition really is what you have.

## 1. Name and place

`<Target>Config`, in the `dlk` module that uses the target. `DataLoaderConfig` lives in `dlk/data/loader.py`. The test goes in the mirrored path under `tests/`.

## 2. Read the target's signature first

```sh
uv run .agents/skills/write-config-dataclass/scripts/check_config.py torch.utils.data.DataLoader
```

This prints every parameter with its annotation and its torch default, which is the list to choose from.

## 3. Stored fields are target parameters, and nothing else

Every stored field is named exactly as a parameter of the target. Nothing else is stored, so `to_kwargs()` stays a one-liner and the test in step 10 works.

Leave out the parameters that belong to the call rather than to the configuration: `dataset`, `sampler`, `collate_fn`, `generator`. Those are passed at the call site.

## 4. Choose one of three kinds per field

- **Required, no default.** The project must decide it every time, so torch's default is dropped on purpose. `shuffle`, `drop_last`, and `batch_size` are required for exactly that reason.
- **Plain field with a default.** A fixed choice that needs no heuristic: `timeout`, `pin_memory_device`, `in_order`.
- **Auto-resolved.** `None` has to mean "decide for me". Write it as a pair: `param_<name>: InitVar[Optional[T]]` for the input, and `<name>: T = field(init=False)` for the resolved value.

Use the third kind only where a heuristic exists. Everything else stays plain.

Give a `param_*` input a default of `None` where the heuristic is the path most callers want, so `AdamWConfig(lr=3e-4)` is enough. Leave it required, as `DataLoaderConfig` does, where every script should have to state what it wants even if that is "decide for me".

Declare fields in this order: required fields, the `param_*` inputs, the `init=False` results, then the plain defaulted fields. Dataclasses require defaults to follow non-defaults, and the order reads as input before result.

## 5. `__post_init__` holds the policy

It takes the `param_*` inputs as arguments and assigns every `init=False` field. The shape is one conditional per parameter:

```python
self.num_workers = (
    param_num_workers if param_num_workers is not None else self.auto_num_workers()
)
```

Gate parameters that only mean something together. `prefetch_factor`, `persistent_workers`, and `multiprocessing_context` are resolved inside `if self.num_workers > 0:`, so a single-process loader keeps the inert values from their `field(init=False, default=...)`. A gated field needs that default, because `__post_init__` may never reach it.

A gate can also be an exclusion. `fused` and `foreach` cannot both be enabled, so resolving `fused` first decides whether `foreach` is resolved at all. Where the caller asks for both explicitly, raise `ValueError` rather than silently dropping one; a heuristic may be overruled quietly, an explicit request may not.

After construction every stored value is concrete. No `None` that still means "undecided" survives, and printing the config shows what the run will actually use.

## 6. Heuristics are public static methods

```python
@staticmethod
def auto_num_workers(
    min_num_workers: int = 1,
    cpu_cores_ratio: float = 0.2,
    cpu_cores_ratio_device: float = 0.5,
) -> int:
```

Every constant the heuristic uses is an argument with a default, so a caller can pin it and a test can drive it. A heuristic written inline in `__post_init__` can be neither.

Probe the machine through torch and the standard library: `torch.accelerator.is_available()`, `os.cpu_count()`. Handle the case where the probe answers nothing, as `os.cpu_count()` returning `None` does.

## 7. Two methods complete the class

**`from_dict`** maps torch's parameter names onto the `param_*` inputs through an explicit alias dict, so configs in YAML, JSON, or CLI flags are written with torch's names and never see the `param_` spelling. Document that it raises `TypeError` on an unknown or missing key.

**`to_kwargs`** returns `{field_.name: getattr(self, field_.name) for field_ in fields(self)}`, annotated `dict[str, Any]`. It never grows a special case; `fields()` already excludes the InitVars. Annotating it `dict[str, object]` type checks on its own and then fails at every `**config.to_kwargs()` call site.

## 8. Several configs in one family share behavior, not fields

`AdamConfig`, `AdamWConfig`, and `SGDConfig` each declare their own fields, because each mirrors a different torch class. What they share, `auto_fused`, `auto_foreach`, `from_dict`, and `to_kwargs`, lives in a plain mixin class that they inherit.

Do not reach for dataclass inheritance to share fields. It forces the base class's fields to come first, which fights the ordering in step 4, and the subclass `__post_init__` has to accept every inherited InitVar in declaration order.

A mixin that calls `fields(self)` needs one line for the type checker, since the mixin itself is no dataclass:

```python
__dataclass_fields__: ClassVar[dict[str, Any]]
```

## 9. Docstrings

Google style, as everywhere in `dlk`. The opening line names the target: "Typed argument set for `torch.utils.data.DataLoader`." Follow it with one sentence listing the fields that fall back to a heuristic when their input is `None`.

`Attributes:` documents every field, including the `param_*` inputs. Each of those reads "Constructor-only input; `None` to ..., (not stored on the instance)", which is what tells a reader why the name has a prefix.

## 10. The test is a drift guard

```python
def test_config_fields_are_valid_dataloader_arguments() -> None:
    dataloader_parameters = inspect.signature(torch.utils.data.DataLoader.__init__).parameters
    config_fields = {field.name for field in dataclasses.fields(DataLoaderConfig)}
    assert config_fields <= dataloader_parameters.keys()
```

This does not test torch. It fails when a torch upgrade renames or removes a parameter the config still stores, which is the failure that would otherwise surface as a `TypeError` in the middle of a training run.

The assertion is a subset, not an equality, because a config covers only the parameters it chooses to cover.

For a family, parametrize it over the `(config, target)` pairs so a new member is one line.

Two more tests earn their place. One pins a heuristic edge, with `monkeypatch` on the probe (`torch.accelerator.is_available`) so the result does not depend on the machine running the suite. One asserts `Config.from_dict(config.to_kwargs()) == config`, which is the round trip every stored preset depends on and which catches an alias map that drifted from the fields.

Do not add a test that constructs the real target to prove the kwargs are accepted; the subset assertion already says that.

## 11. Check the result

```sh
uv run .agents/skills/write-config-dataclass/scripts/check_config.py torch.utils.data.DataLoader dlk.data.loader.DataLoaderConfig
```

It prints the target's parameters, then the config's fields with their kind, then its findings. It fails when a stored field is not a parameter of the target, when a constructor-only input lacks the `param_` prefix or pairs with no stored field, when a field set from a `param_*` input forgot `init=False`, and when an `init=False` field has no input to set it.

Its `INFO` lines are choices to confirm, not errors: the target parameters left uncovered, and the fields whose default departs from torch's. Every one of those should be deliberate.

Then run `make format`, `make lint`, and `make test`.

## Pitfalls

- A `param_*` value kept on the instance defeats the design. The `InitVar` exists so the unresolved input cannot be read back later.
- Ordering errors surface as a confusing dataclass `TypeError` about a non-default argument following a default one. Re-read step 4 rather than sprinkling defaults.
- Changing a heuristic changes every run that relies on it, silently. That is why its constants are arguments: a caller who needs the old behavior pins it.
- `dataclasses.fields()` returns stored fields only, never InitVars. Both the test and `to_kwargs` depend on that.
- A heuristic that reads only `torch.accelerator.is_available()` cannot see where the tensors are. `auto_fused` returns True on a machine with an accelerator whose model still sits on the CPU, and the optimizer then raises at construction. Where a heuristic has a precondition the config cannot check, say so in its docstring and name the way out (`param_fused=False`).
- A legacy dict API that renames or combines parameters, such as `beta1` and `beta2` into `betas`, does not belong in `from_dict`. Keep `from_dict` a pure alias map over the target's own names and leave the translation in the wrapper function that serves the old callers.
