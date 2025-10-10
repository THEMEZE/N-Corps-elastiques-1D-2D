# 🧠 N-Corps-elastiques-1D-2D

---

## ⚙️ *Lancer Jupyter Notebook depuis le répertoire d’une clé USB* en utilisant soit un **venv Python** soit **conda**, et **installer les dépendances depuis `requirements.txt`**.

> Supposons que ta clé est montée sur `/Volumes/NO NAME` et que ton projet est dans `/Volumes/NO NAME/N_Corps_elastiques_1D_2D`.


### Option A — méthode simple avec **python - venv** (macOS / Linux / Terminal zsh)

1. (recommandé) créer le venv sur le **disque local** (rapide & fiable) puis pointer Jupyter sur la clé :

```bash
# créer venv dans ton home (ou .venvs)
python3 -m venv ~/venvs/N_Corps_elastiques_1D_2D_env
# activer
source ~/venvs/N_Corps_elastiques_1D_2D_env/bin/activate
# mettre pip à jour
pip install --upgrade pip
# installer les dépendances
pip install -r requirements.txt
# installer jupyter et ipykernel si besoin
pip install jupyter ipykernel
# enregistrer le kernel (facultatif mais pratique)
python -m ipykernel install --user --name N_Corps_elastiques_1D_2D_env --display-name "Python (N_Corps_elastiques_1D_2D_env)"
```

2. lancer Jupyter Notebook **dans le dossier de la clé** :
```bash
# méthode 1: lancer depuis le répertoire ouvert (si tu as cd dans /Volumes/NO NAME/N_Corps_elastiques_1D_2D)
jupyter notebook

# ou méthode 2: lancer explicitement en pointant le dossier de la clé
jupyter notebook --notebook-dir="/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
```

> Ensuite dans l’interface web, choisis le kernel `Python (N_Corps_elastiques_1D_2D_env)` si tu as enregistré le kernel.


### Option B — avec conda (si tu utilises Anaconda / Miniconda)

1. depuis le terminal :
```bash
cd "/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
# créer l'env (ici on installe via pip requirements.txt)
conda create -n N_Corps_elastiques_1D_2D_env python=3.11 -y
conda activate N_Corps_elastiques_1D_2D_env
# installer pip si besoin
conda install pip -y
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name N_Corps_elastiques_1D_2D_env --display-name "Conda (N_Corps_elastiques_1D_2D_env)"
# lancer jupyter
jupyter notebook --notebook-dir="/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
```

Si tu as un environment.yml tu peux faire :
```bash
conda env create -f environment.yml -n N_Corps_elastiques_1D_2D_env
conda activate N_Corps_elastiques_1D_2D_env
```


### Option C — si tu veux éviter d’installer un env : lancer Jupyter avec `python -m` depuis la clé (moins recommandé)
Si tu as déjà Python sur la clé (rare), ou si tu veux utiliser l’interpréteur système :
```bash
cd "/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
python3 -m pip install --user -r requirements.txt   # installe dans ~/.local
python3 -m notebook --notebook-dir="."
```

### Cas particulier : tu veux conserver l’environnement sur la clé malgré les risques

Si tu veux absolument créer le venv sur la clé :

```bash
cd "/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook --notebook-dir="."
```
> [!WARNING]
> ⚠️ Si la clé est formatée en FAT32, certains exécutables/script pourront échouer (pas de permission d’exécution, pas de liens symboliques). Si tu as des erreurs `Read-only file system` ou perte de permissions, reformatte la clé en exFAT ou APFS (si tu peux) ou utilise l’option recommandée : venv local.

### Résolution des problèmes courants

- **Permission denied / Read-only file system :** la clé est montée en lecture seule ; démonte et remonte en écriture ou vérifie que la clé n’est pas verrouillée physiquement.

- **Nom de volume avec espaces :** entoure le chemin de guillemets comme ci-dessus `"/Volumes/NO NAME/N_Corps_elastiques_1D_2D"`.

- **Clé lente / accès disque lent :** créer le venv local + travailler sur la clé (notebooks sur la clé) ou copier le repo en local.

- **Kernel absent dans Jupyter :** installe `ipykernel` et exécute `python -m ipykernel install --user --name ....`

- **requirements.txt incomplet / erreur pip :** regarde les lignes d’erreur, certaines bibliothèques peuvent nécessiter des dépendances système (Xcode command line tools, libffi, etc.).

- **Utiliser JupyterLab :** remplace `jupyter notebook` par `jupyter lab` si tu préfères l’interface moderne (installer `jupyterlab`).

### Exemple complet (copier-coller, venv sur le home, clé `/Volumes/NO NAME/N_Corps_elastiques_1D_2D`)
```bash
# Terminal (macOS)
cd "/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
python3 -m venv ~/venvs/N_Corps_elastiques_1D_2D_env
source ~/venvs/N_Corps_elastiques_1D_2D_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name N_Corps_elastiques_1D_2D_env --display-name "Python (N_Corps_elastiques_1D_2D_env)"
jupyter notebook --notebook-dir="/Volumes/NO NAME/N_Corps_elastiques_1D_2D"

```
---

### ⚙️ Git Mise à jour
```bash
git add .
git commit -m "Mise à jour"
git push
```

