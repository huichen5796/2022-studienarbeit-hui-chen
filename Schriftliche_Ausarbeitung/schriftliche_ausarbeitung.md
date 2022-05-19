# Schriftliche Ausarbeitung

## Titelseite

**Projektarbeit**

*Automatisierte Erfassung und Extraktion von Daten aus Tabellen in Dokumenten.*

Name: Hui Chen
Matr.-Nr.: 10048521

## Einleitung

## Grundlagen

- Daten, Informationen, Wissen (--> Pyramide)
- Tabellen
- Python

![Pyramide](_images/wissenspyramide_derwirtschaftsinformatiker.png)

## Stand der Technik

- [OpenCV][1]
- Tesseract

## Implementierung

### Idee & Planung

- mit idealem Bild starten und wenn Extraktion erfolgreich ist, auch nicht-optimale Bilder erproben

#### Ablauf

- ggf. Korrektur von Fehlern (z.B. tilt, Artefakte)
- Binärisierung
- Detektion von Zellen
- OCR auf Zelle
  - Oberste Zeile (header) --> keys (= Feldnamen; z.B. "Datum", "Name", ...)
  - Zeilen darunter --> values (= Werte; z.B. "12-02-22", "Mustermann", ...)
- Erstellung von Datenobjekten je Zeile mit keys aus header
- Speicherung von Datenobjekten in Datenbank (Elasticsearch)

### Programmierung

### Test

#### Binarization

![binarization](_images/binarization_test.png)

## Diskussion

## Zusammenfassung

## Ausblick

## Quellen

- [1]: <https://de.wikipedia.org/wiki/OpenCV>
