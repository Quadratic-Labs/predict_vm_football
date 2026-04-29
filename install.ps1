# install.ps1

# Stop en cas d'erreur
$ErrorActionPreference = "Stop"

# Vérifie Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python n'est pas installé"
    exit 1
}

# Crée venv si absent
if (-not (Test-Path "venv")) {
    Write-Host "Création de l'environnement virtuel..."
    python -m venv venv
}

# Active venv
Write-Host "Activation de l'environnement virtuel..."
.\venv\Scripts\Activate.ps1

# Update pip
Write-Host "Mise à jour de pip..."
python -m pip install --upgrade pip

# Installe requirements
if (Test-Path "requirements.txt") {
    Write-Host "Installation des dépendances..."
    pip install -r requirements.txt
} else {
    Write-Host "requirements.txt introuvable"
    exit 1
}

Write-Host "Installation terminée"