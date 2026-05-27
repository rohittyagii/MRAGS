This folder contains small examples and scripts to demonstrate a typical local run of MRAGS.

Place a PDF named `sample.pdf` in this folder before running the demo scripts, or update the paths in the commands.

Windows PowerShell quick start

1. Create a venv and install dependencies (from repo root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt  # or `pip install .` for editable install
```

2. Ensure your local model path and other settings are set in `.env` or as environment variables.

3. Run the demo script:

```powershell
# validate LMM
python -m mrags.cli validate-lmm

# ingest the sample PDF
python -m mrags.cli ingest examples\sample.pdf

# show retrieved context only (no generation)
python -m mrags.cli query "Give a one-sentence summary." -c

# generate an answer using LMM
python -m mrags.cli query "Give a one-sentence summary."

# inspect index and DB
python -m mrags.cli status
```

Python demo runner

You can also run `examples/run_demo.py` directly to execute the same steps programmatically (it will call into the CLI helpers). Replace `examples/sample.pdf` with your PDF path as needed.
