# Automatische Erkennung und Konvertierung von Tabellen in Bilddokumenten mit Hilfe von Machine Learning

## About this

Die ist ein Tool zur...

- Erkennung
- Extraktion
- Rekonstruktion

... von komplexen Tabellen aus Bilddokumenten.

![principle](Abbildungen\ablauf.gif)

Es basiert auf:

- Python
- Tesseract
- ... vervollständigen!!!

Es wurde für *Windows* entwickelt, lässt sich aber auch auf anderen Betriebssystemen zum Laufen bringen.
Das Tool besteht aus zwei wesentlichen Teilen:

1. Training von neuronalen Netzen mittels *Torch*. Dieses Training wird idealerweise auf einer rechenstarken Maschine ausgeführt (z.B. *Google Collab* mit GPUs).

2. Erkennung von Tabellen in Bild- oder PDF-Dokumenten auf Basis des zuvor trainierten neuronalen Netzes.

## Setup

Um beide Programmbausteine lauffähig zu machen, müssen folgende Schritte ausgreführt werden:

- Installation von *Python*
- Installation von *Python*-Paketen
- ...

### Installation von *Python* und Paketen

- Letzte Version von *Python* [hier](https://www.python.org/ftp/python/) herunterladen und installieren.

- Geforderte Pakete installieren: `pip install -r requirements.txt`

### Installation von *Elasticsearch*

- Elasticsearch kann nach folgender Anleitung installiert werden: [Installation Elasticsearch](https://youtu.be/Tn6zkPz-qHc?t=553)

1. *Elasticsearch* [hier](https://www.elastic.co/de/downloads/elasticsearch) herunterladen (getestet für *Version 7.17.1*).
2. Archiv entpacken (z.B. nach `D:\elasticsearch\`)
3. Navigation in den Ordner `elasticsearch\bin`
4. `elasticsearch.bat` ausführen, um die Installation zu starten.
5. `localhost:9200` im Browser eingeben, um erfolgreiche Installation zu testen. &rarr; Folgender Text sollte im Browser lesbar sein: "You know you search."

**Hinweis:** Die Version des pip-Packages für *Elasticsearch* muss zur installierten Version auf dem System passen!

### Installation von *Tesseract* (für Windows)

- Anleitung zur Installation: [*Installing and using Tesseract 4 on windows 10*](https://medium.com/quantrium-tech/installing-and-using-tesseract-4-on-windows-10-4f7930313f82)
- Installationsdateien [hier](https://github.com/UB-Mannheim/tesseract/wiki) runterladen (`tesseract-ocr-w64-setup-$VERSION$.exe`) und ausführen.
- "Additional script data (doiwnload)" und "Additional language data (download)" auswählen.

![alle Sprachen](Abbildungen\installtesse.jpg)

- Installationspfad wählen.
- Installationspfad zu Systemumgebungsvariable `PATH` hinzufügen.

![Sysvars](Abbildungen\systemumgebungsvariablen.png)

![Pfad hinzufügen](Abbildungen\zupathadd.jpg)

- Neue Systemvariable erstellen:
  - Variablename: `TESSDATA_PREFIX`
  - Variablenwert ist Installationspfad, z.B. `C:\Program Files\Tesseract-OCR\tessdata`

![neue Systemvaiable](Abbildungen\tesserdata.jpg)

### Installation von *pytorch* 

- Command for Installation von Pytorch wird hier bekommen: [Get Started of Pytorch](https://pytorch.org/get-started/locally/) -->

- Ihre Einstellungen auswählen und den Installationscommand durchführen.

![pytorch_command](Abbildungen\pytorch.jpg)

- den Command in Terminal kopieren und durchführen.

wie zum Beispiel: 
`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`

## Programmablauf

Der Ablauf des Programmes kann anhand den Folgenden  nachvollzogen werden:

- Die Verarbeitung einzeles Bilds

![verarbeitung einbild](Abbildungen\programmablauf.svg)

- Stapelverarbeitung mehrer Bilder

![stapelverarbeitung einbild](Abbildungen\stapelverarbeitung.svg)


## Ergebnisse

- Die Verarbeitung einzeles Bilds
  - Vorbreitung und Normalizierung

  ![vorbreitung](Abbildungen\vorverarbeitung.png)
   

  - Erkennung des Tablebreichs

  ![erkennung table](Abbildungen\erkennung.png)

  - Erkennung der Zelle

  ![erkennung cells](Abbildungen\cell.png)

  - Rekonstruktion
    - Columen Detection mittels ML Modell, somit werden Labels von Columen erstellt.

    ![ml for cols](Development\imageSave\table_1_of_test3.png)

      (Die rote Linie ist die Mittellinie der durch maschinelles Lernen erkannten Tabellenspalte, und die Zellen, die sich auf beiden Seiten der roten Linie innerhalb der grünen Linien befinden, werden in einer Spalte gruppiert.)

    - Zuweisung der Labels

      Labels werden anhand Positon von jeder Zelle erstellt.

      ![labels](Abbildungen\labels.jpg)

    - Rekonstruktion

      ![wiederaufbau](Abbildungen\table.jpg)

    - Strukturnormalize
      - vertikales Schmelzen von zwei geschmelzbaren Zeilen
      - horizonales Schmelzen von der ersten und zweiten Spalte
      - Bestimmen der Zeilennummer von header
      - Bestimmen der Qualifikationen der Zellen in erster Zeile, dilatiert zu werden und ausgefüllt zu werden
      - Verarbeitung der erster Zeile
      - Schmelzen von header

      ![umform](Abbildungen\umform.gif)

- Stapelverarbeitung mehrer Bilder

  - Die Bilder im Verzeichnis werden zuerst formatiert, alle PDFs werden Seite für Seite in das PNG-Dateiformat konvertiert.

  ![stapel](Abbildungen\stapel_vor.jpg)

  - Dann wird jedes Bild verarbeitet und in Elasticsearch geschrieben.

  ![einschreiben](Abbildungen\stapelverarbeitung.jpg)

- Leistung bei komplexer Tabelle

![1](Development\imageTest\test2.PNG)

![2](Abbildungen\komplexbild.jpg)

  nach Strukturnormalize:

![3](Abbildungen\sn.jpg)

