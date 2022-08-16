# Schriftliche Ausarbeitung

## Titelseite

**Projektarbeit**

*Automatisierte Erfassung und Extraktion von Daten aus Tabellen in Dokumenten.*

Name: Hui Chen
Matr.-Nr.: 10048521
## Abstrakt
- Problematik 
    - freie automatische Erfassung der Tabellen in Bildern
- Thema/Ziel
    - oben genannt
- Vorgehen/Methodik
    - Kooperation von traditionellen Methoden und ML
- Wichtigste Ergebnisse
    - brauch Bewertungsindikatoren für Genauigkeit von Wiederaufbau der Tabellenstruktur und von der Erfassung des Inhalts
- Keywords

## Einleitung
- Kontext, Zusammenhang mit anderen Arbeiten ?
- Problemstellung
- Ziel
- Überblick der Arbeit, hierbei kurz die Vorgehensweise


![Pyramide](./_images/wissenspyramide_derwirtschaftsinformatiker.png)

## Stand der Technik
### Grundlagen der Datenverarbeitung
- Daten, Informationen, Wissen (--> Pyramide)
- Tabellen

### Traditionelle digitale Bildverarbeitung
- grundliegende Erkenntnisse
- [OpenCV][1] sowie ggf. Matlab
- Objekt Detection mittels tradition. DB 

### Maschinen Learning
- Grundlagen Maschinen Learning
- Classifizieren, Objekt Detection, Sematische Sigmentation, hierbei einige bekannte Modelle nennen sollten.
- OCR basierende auf ML z.B. Tesseract
- GOOGLE COLAB

## Implementierung

### Idee & Planung
- mit idealem Bild starten und wenn Extraktion erfolgreich ist, auch nicht-optimale Bilder erproben
- mehr Robustness durch ML, mehr Genauigkeit durch traditionelle DB

#### Vergleichung Methoden Binärization
- 'Grundlagen_Erkenntinesse\Traditionell\functionsDebug\Binarization.py'

#### Vergleichung Methoden von TiltCorrection
- hierbei Vergleichung Methoden von Linienerkennung
    - issue 'LinienErkennung HOUGH, LSD, FLD'
- issue 'Bug bei TiltCorrection'

#### Detection von Tabellenbreich
- Vergleichung traditionell DB(Linienerkennung) und ML
- Einführung von (eine von Unet oder Tablenet(also wesentlich VGG19) oder Densenet)
    - Modellstruktur
    - Ergebinisse, also loss, acc, runtime in CPU

#### Vergleichung Methoden Detektion von Zellen
- Wiederaufbau aller Linien dann Erkennung der Zellen mittels Erkennung der Linien
- Delete aller Linien dann direkte Erkennung der Textblock in jeder Zellen

#### Analyse der Tabellenstruktur und normalizierte Einschreibung in Elasticsearch
- Vorgehensweise see isse 'instraction to funciton Umform()'
    - hierbei auch PositionCorrection see issue 'PositionCorrection'
    - hierbei ist ML genutzt für columen detection
        - Ergebinisse, also loss, acc, runtime in CPU

#### Ablauf

- ggf. Korrektur von Fehlern (z.B. tilt, Artefakte)
- Normalize der Bilder
- Detektion von Tabellenbreich
- Analyse der Tabellenstruktur
- Detektion von Zellen
- OCR auf Zelle
  - Oberste Zeile (header) --> keys (= Feldnamen; z.B. "Datum", "Name", ...)
  - Zeilen darunter --> values (= Werte; z.B. "12-02-22", "Mustermann", ...)
- Erstellung von Datenobjekten je Zeile mit keys aus header
- Speicherung von Datenobjekten in Datenbank (Elasticsearch)

#### Funktionsarchitektur

- es gibt zwei Funktionsarchitekturen/Programme:
  - Training
  - Produktiver Einsatz

##### Training

- Bilder und Masken werden manuell bereitgestellt
- Modell wird durch Vorgaben des Menschen trainiert
- Am Ende steht ein fertiges Modell, dass auch ohne Training also "offline" genutzt werden kann.

##### Produktiver Einsatz

- Das Programm für den produktiven Einsatz nutzt das trainierte Modell

- `extractTableDataFromPdfFile(pdfFilePath)`
  - `pdfFileToPngFiles(pdfFilePath, tempFolderPath)` (return: void)
  - `detectTablesInPngFiles(tempFolderPath)` (return: void)
    - `detectTableInPngFile(pngFilePath)` (return: {xPositionTable, yPositionTable}) &larr; hier wird das trainierte Modell genutzt!
    - `extractTableDataFromPngFile(pngFilePath, xPostionTable, yPositionTable)` (return: {headerData, rowData[]})

![komplexe Tabelle](./_images/prozess_zur_extraktion_von_tabellen.drawio.svg)

### Programmierung

#### Tabellentypen

![einfache Tabelle](./_images/tabellentypen-einfach.drawio.svg)

![komplexe Tabelle](./_images/tabellentypen-komplex.drawio.svg)

#### Binarization

![binarization](./_images/binarization_test.png)

## Diskussion
- Normalerweise besteht keine starke Korrelation zwischen den beiden Tabellen, daher werden sie als separate Blöcke in das Verzeichnis geschrieben.
- Um die Tabellenstruktur möglichst nicht zu zerstören, werden ,,Header" und ,,Wert" auf gleicher Ebene in Elasticsearch geschrieben.

## Zusammenfassung
- In dieser Studienarbeit werden eine Methode zur Automatisierte Erfassung und Extraktion von Daten aus Tabellen bringen, mittels Maschinen Learing gibt es bei der Erkennung von Tablebreich mehr Robustness, und mittels traditioneller Verfahrensweisen hat die Rekonstruktion mehr Genauigkeit.


## Ausblick
- Genauigkeit der Modell
- die Auswahl einiger Schwellen bei traditionell DB hat ggf. keinen genugen Robustness
- Analyse der Struktur kann durch ML durchführen


## Quellen

- [1]: <https://de.wikipedia.org/wiki/OpenCV>
