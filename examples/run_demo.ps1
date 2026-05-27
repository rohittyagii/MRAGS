# Quick PowerShell demo for MRAGS
# Put a PDF named 'sample.pdf' in the examples/ folder or update the path below.

# Optional: load .env variables (requires 'Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass' if blocked)
# Get-Content .env | Foreach-Object { if ($_ -and -not $_.StartsWith('#')) { $parts = $_ -split '='; $env[$parts[0]] = $parts[1] } }

python -m mrags.cli validate-lmm
python -m mrags.cli ingest examples\sample.pdf
python -m mrags.cli query "Give a short summary." -c
python -m mrags.cli query "Give a short summary."
python -m mrags.cli status
