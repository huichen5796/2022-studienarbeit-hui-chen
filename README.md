# 2022-projektarbeit-hui-chen

- Hier ist der Arbeitsplatz für die Studienarbeit “Automatische Erkennung und Konvertierung von Tabellen in Bilddokumenten”.

## TODOS

- Nachdem die Arbeit endet, sollte hier aufgeräumt werden.

## Setup

- Install Python & pip
- Install OpenCV package for Python: `pip install opencv-python`
- Tesseract für Windows:
  - Information für [Download](<https://github.com/tesseract-ocr/tessdoc/blob/main/Downloads.md>)
  - auf [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) klicken
  - die entsprechende Version herunterladen
  - Tesseract-OCR installieren

    ![install](Schriftliche-Ausarbeitung\\_images\\install0.jpg)
  - den Installationspfad wählen
    
    ![install1](Schriftliche-Ausarbeitung\\_images\install1.jpg)
  - zu den Systemvariablen (PATH) der Umgebungsvariablen hinzufügen

    ![zu PATH](Schriftliche-Ausarbeitung\\_images\zupath.jpg)
    
  - eine neue Systemvariable erstellen
    - Variablename: TESSDATA_PREFIX-Variablennamen 
    - Variablenwert ist Installationspfad, z.B. C:\Program Files\Tesseract-OCR\tessdata

    ![tessdata](Schriftliche-Ausarbeitung\\_images\tessdata.jpg)
  - in der Datei _pytesseract.py_ (unter dem Pfad, den Sie gerade installiert haben) ändern `tesseract_cmd = 'tesseract'`  in `tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe"`
    
    oder: einfach in python-code `pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'` nutzen
  - Install Tesseract package for Python: `pip install pytesseract`