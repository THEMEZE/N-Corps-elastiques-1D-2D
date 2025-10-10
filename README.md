# 🧠 N-Corps-elastiques-1D-2D

---

## ⚙️ *Lancer Jupyter Notebook depuis le répertoire d’une clé USB* en utilisant soit un **venv Python** soit **conda**, et **installer les dépendances depuis `requirements.txt`**.

> Supposons que ta clé est montée sur `/Volumes/NO NAME` et que ton projet est dans `/Volumes/NO NAME/N_Corps_elastiques_1D_2D`.

### Option A — méthode simple avec **python - venv** (macOS / Linux / Terminal zsh)

1- (recommandé) créer le venv sur le **disque local** (rapide & fiable) puis pointer Jupyter sur la clé :

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

2- lancer Jupyter Notebook **dans le dossier de la clé** :
```bash
# méthode 1: lancer depuis le répertoire ouvert (si tu as cd dans /Volumes/NO NAME/mon_projet)
jupyter notebook

# ou méthode 2: lancer explicitement en pointant le dossier de la clé
jupyter notebook --notebook-dir="/Volumes/NO NAME/N_Corps_elastiques_1D_2D"
```

---

### ⚙️ Git Mise à jour
```bash
git add .
git commit -m "Mise à jour"
git push
```

