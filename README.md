# Metalinguistic Features in Political Bias Identification

Does giving a language model the *metalinguistic* signals around a news article —
what it is about, and how it feels about it — help it recognise political bias?

This repo tests that in two stages:

1. **Pretraining.** Continue training RoBERTa-base on a large news corpus with four
   objectives at once: masked language modelling, a contrastive objective that pulls
   same-leaning articles together, multi-label prediction of the article's GDELT
   themes, and regression on its GDELT tone.
2. **Fine-tuning.** Take that checkpoint, fine-tune it on labelled bias datasets
   (AllSides and friends), and compare it against off-the-shelf BERT, BART, RoBERTa
   and POLITICS.

---

## Setup

Python 3.12. Install torch first, because the right wheel depends on your machine:

```bash
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

## Stage 1: pretraining

Every run is described by one YAML file in `run_configs/`.

Set your accelerate configurations by running:
```bash
accelerate config
```

```bash
accelerate launch pretraining.py --config run_configs/tlp_tone_16.yaml
```
To queue every config in
`run_configs/` back to back:

```bash
./run_all.sh
```

### Writing a run config

```yaml
output_dir: ./my_run            # everything this run produces lands here
tasks: [triplet, mlm, tone]     # which objectives to train; omit any you don't want
theme_count: 2000               # size of the theme label space (top_themes.txt)
base_lr: 5.0e-5
train_args:
  num_epochs: 1
  batch_size: 32                # per GPU
  warmup_ratio: 0.06
  log_every: 500                # steps between progress lines
task_spec:
  dataset_name: upasanachatterjee/bignewsalign-with-gdelt
  themes_path: top_themes.txt
  max_triplet_samples: 16       # triplets mined per batch
  require_nonempty_themes_and_tone: True
```

Two optional blocks:

```yaml
init_from_checkpoint: ./my_run/epoch-1.pt   # continue from an earlier run
loss_weights:                               # default is an unweighted sum
  triplet: 1.0
  themes: 1.0
  tone: 1.0
  mlm: 1.0
```

`init_from_checkpoint` reads the epoch number out of the filename and continues the
numbering, so pointing a new run at `epoch-1.pt` with the same `output_dir` writes
`epoch-2.pt` rather than overwriting anything.

---

## Stage 2: fine-tuning and evaluation

Open the notebook:

```bash
jupyter notebook run-finetuning-experiments.ipynb
```

Run the cells top to bottom. It downloads the published checkpoint from the Hub,
then sweeps it and the four baselines over the media-split and random-split versions
of the AllSides data:

```python
media_results = run_all_models(
    dataset=ALLSIDES_EXTENDED_MEDIA_SPLIT,
    output_prefix="results_undersampling/media_split",
)
```

Each model writes a `<name>_baseline_trunc_test_metrics.json` into
`output_prefix`, containing accuracy, macro/micro F1, precision and recall, the
per-class breakdown, the raw predictions, and the training arguments used.

To evaluate a checkpoint you trained yourself, pass its path instead of a hub name:

```python
run_experiment(
    model="./my_run/epoch-1.pt",
    loc="results/my_run",
    dataset_config=DatasetConfig(custom_dataset=ALLSIDES_EXTENDED_MEDIA_SPLIT),
    experiment_config=ExperimentConfig(patience=3, num_epochs=15),
)
```

The bias head is new at this point — pretraining never trains one — so it starts from
random weights and you'll see a "missing keys" count. That's expected.

**Media split vs random split.** The media split keeps every outlet entirely within
one of train/test, so a model can't succeed by memorising outlets. It is the harder
and more meaningful comparison; the random split is reported alongside it.

**Undersampling.** The training split is always balanced per topic.