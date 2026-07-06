import cv2
from pathlib import Path

input_folder = Path("predictions/input")
output_folder = Path("predictions/input")

output_folder.mkdir(exist_ok=True)

image_files = (
    list(input_folder.glob("*.jpg")) +
    list(input_folder.glob("*.jpeg")) +
    list(input_folder.glob("*.png"))
)

TARGET_SIZE = (432, 432)

for img_path in image_files:

    img = cv2.imread(str(img_path))

    if img is None:
        print(f"Fehler beim Laden: {img_path}")
        continue

    # RGB → Graustufen
    img_gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Größe anpassen
    img_resized = cv2.resize(
        img_gray,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # PNG-Dateiname erzeugen
    new_name = img_path.stem + ".png"

    save_path = output_folder / new_name

    cv2.imwrite(
        str(save_path),
        img_resized
    )

    print(
        f"{img_path.name} → {new_name}"
    )

print("Fertig")