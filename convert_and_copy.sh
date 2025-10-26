#!/bin/bash

# Dossiers
INPUT_DIR="./outputs"
OUTPUT_DIR="./vitrines"

# Crée le dossier vitrines s'il n'existe pas
mkdir -p "$OUTPUT_DIR"

# Liste des fichiers à convertir
declare -a FILES=(
  "2D_positions_animation"
  "2D_velocity_distribution"
  "1D_positions_animation"
  "1D_velocity_distribution"
)

# Paramètres communs
FPS=60
SCALE=800

echo "🎞️ Conversion des fichiers MP4 en GIF..."
for NAME in "${FILES[@]}"; do
    MP4_PATH="$INPUT_DIR/${NAME}.mp4"
    GIF_PATH="$INPUT_DIR/${NAME}.gif"
    FINAL_PATH="$OUTPUT_DIR/${NAME}.gif"

    if [ -f "$MP4_PATH" ]; then
        echo "➡️  Conversion de $MP4_PATH ..."
        ffmpeg -y -i "$MP4_PATH" -vf "fps=$FPS,scale=${SCALE}:-1:flags=lanczos" "$GIF_PATH"

        echo "📂 Copie de $GIF_PATH vers $FINAL_PATH ..."
        cp -f "$GIF_PATH" "$FINAL_PATH"
    else
        echo "⚠️  Fichier manquant : $MP4_PATH"
    fi
done

echo "✅ Conversion et copie terminées !"

