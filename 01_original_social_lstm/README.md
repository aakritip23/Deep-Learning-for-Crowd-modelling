# Reference implementation: Social LSTM (course-provided baseline)

`social-lstm/` is a copy of <https://github.com/quancore/social-lstm> (PyTorch implementation of Social LSTM, Alahi et al., CVPR 2016),
the baseline provided for the course. Upstream commit: `bdddf36e312c9dde7a5f1d447000e59918346520` (2020-04-26).
Its own README is [`social-lstm/README.md`](social-lstm/README.md); it quotes the published TrajNet test-server result that this project tries to beat:
**Social LSTM Avg 1.3865 = (FDE 2.098 + ADE 0.675) / 2**.

## Differences to upstream
Only three files differ from the upstream commit: `model.py`, `train.py` and `utils.py`
(full diff: [`local_changes_vs_upstream.patch`](local_changes_vs_upstream.patch), it applies cleanly to the upstream commit). The changes were made while trying to
run the baseline on Windows:

* `train.py` - output directories are created with `os.makedirs` instead of calling `make_directories.sh`; `is` / `is not` comparisons with numbers replaced by `==` / `!=`;
  the grid masks are computed per sequence instead of being cached across epochs; the training forward pass uses `x_seq[:-1]` with next-step targets `x_seq[1:]`;
  the validation loss is computed against the shifted ground-truth sequence.
* `utils.py` - `is` -> `==`; dataset folder names are derived with `os.path` instead of splitting paths on `/`; `get_all_directory_namelist` lists the sub-folders of the scanned directory; one `pd.unique` call simplified.
* `model.py` - the length of the forward pass is taken from the input tensor (`input_data.size(0)`) instead of `self.seq_length`.

These edits were **not** validated against the published results and the baseline was **not** re-trained for this project (a local smoke test ran about 1.5 s per batch, roughly 2 hours per epoch,
and ran out of memory in the first epoch). The baseline score used throughout the project is the published one; the comparison under identical local conditions uses a
re-implementation of the baseline inside the project's own pipeline (`../03_improved_model/trajpred/reference_lstm.py`, see `../03_improved_model/RESULTS.md`, Section 1).
All other upstream files are unchanged (the unchanged upstream code still prints `SyntaxWarning: "is" with 'int' literal` on Python 3.8+; harmless).

## Data
`social-lstm/data/{train,validation,test}/<source>/*.txt` are the TrajNet text files shipped with the baseline: 131 files, 17.5 MB, sources `biwi` (ETH / Hotel), `crowds` (UCY),
`mot` (PETS09) and `stanford` (Stanford Drone Dataset). One row per observation: `frame  pedestrian_id  x  y` (metres); every trajectory has 20 points (8 observed + 12 to predict).
The test files have hidden ground truth (`?` in the last 12 steps of each trajectory). The data are the input of `../02_data` and `../03_improved_model`.
The baseline's pre-processing caches (`*.cpkl`, about 357 MB) are not part of the repository; the baseline rebuilds them automatically.
