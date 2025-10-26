#!/bin/bash

# Répertoire source et destination
INPUT_DIR="./outputs"
DEST_DIR="/Users/themezeguillaume/Documents/Presentation-These/figures"

# Crée le dossier de destination s’il n’existe pas
mkdir -p "$DEST_DIR"

# Liste des vidéos à traiter
declare -a VIDEOS=(
  "2D_positions_animation"
  "2D_velocity_distribution"
  "1D_positions_animation"
  "1D_velocity_distribution"
)

FPS=60

echo "🎞️ Extraction des frames à $FPS fps..."

for NAME in "${VIDEOS[@]}"; do
    INPUT="$INPUT_DIR/${NAME}.mp4"
    OUTPUT_DIR="$INPUT_DIR/frames_${NAME}"

    if [ -f "$INPUT" ]; then
        echo "➡️  Traitement de $INPUT ..."
        mkdir -p "$OUTPUT_DIR"
        ffmpeg -y -i "$INPUT" -vf "fps=$FPS" "$OUTPUT_DIR/frame_%05d.png"

        # Copie vers le dossier de présentation
        echo "📂 Copie de $OUTPUT_DIR vers $DEST_DIR ..."
        rsync -a --delete "$OUTPUT_DIR" "$DEST_DIR/"
    else
        echo "⚠️  Fichier manquant : $INPUT"
    fi
done

echo "✅ Extraction et copie terminées !"
