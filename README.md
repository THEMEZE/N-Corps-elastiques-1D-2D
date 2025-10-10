# 🧠 N-Corps-elastiques-1D-2D

---

## ⚙️ *Lancer Jupyter Notebook depuis le répertoire d’une clé USB* en utilisant soit un **venv Python** soit **conda**, et **installer les dépendances depuis `requirements.txt`**.

> Supposons que ta clé est montée sur `/Volumes/NO NAME` et que ton projet est dans `/Volumes/NO NAME/N_Corps_elastiques_1D_2D`.

----

### Option A — méthode simple avec **python - venv** (macOS / Linux / Terminal zsh)

1. (recommandé) créer le venv sur le **disque local** (rapide & fiable) puis pointer Jupyter sur la clé :

```bash
# créer venv dans ton home (ou .venvs)
python3 -m venv ~/venvs/mon_projet_env
# activer
source ~/venvs/mon_projet_env/bin/activate
# mettre pip à jour
pip install --upgrade pip
# installer les dépendances
pip install -r requirements.txt
# installer jupyter et ipykernel si besoin
pip install jupyter ipykernel
# enregistrer le kernel (facultatif mais pratique)
python -m ipykernel install --user --name mon_projet_env --display-name "Python (mon_projet_env)"
```

2. lancer Jupyter Notebook **dans le dossier de la clé** :
```bash
# méthode 1: lancer depuis le répertoire ouvert (si tu as cd dans /Volumes/NO NAME/mon_projet)
jupyter notebook

# ou méthode 2: lancer explicitement en pointant le dossier de la clé
jupyter notebook --notebook-dir="/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
```

> Ensuite dans l’interface web, choisis le kernel `Python (mon_projet_env)` si tu as enregistré le kernel.

----

### Option B — avec conda (si tu utilises Anaconda / Miniconda)

1. depuis le terminal :
```bash
cd "/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
# créer l'env (ici on installe via pip requirements.txt)
conda create -n mon_projet_env python=3.11 -y
conda activate mon_projet_env
# installer pip si besoin
conda install pip -y
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name mon_projet_env --display-name "Conda (mon_projet_env)"
# lancer jupyter
jupyter notebook --notebook-dir="/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
```

Si tu as un environment.yml tu peux faire :
```bash
conda env create -f environment.yml -n mon_projet_env
conda activate mon_projet_env
```

----

### Option C — si tu veux éviter d’installer un env : lancer Jupyter avec `python -m` depuis la clé (moins recommandé)
Si tu as déjà Python sur la clé (rare), ou si tu veux utiliser l’interpréteur système :
```bash
cd "/Volumes/NO NAME/mon_projet"
python3 -m pip install --user -r requirements.txt   # installe dans ~/.local
python3 -m notebook --notebook-dir="."
```
---

### ⚙️ Git Mise à jour
```bash
git add .
git commit -m "Mise à jour"
git push
```

