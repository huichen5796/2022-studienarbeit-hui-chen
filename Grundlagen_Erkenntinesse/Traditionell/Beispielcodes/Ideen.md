 Die Ideen über die Arbeit werden hier gespeichert.
## TODOS
- Nun: relevante Abhandlungen lesen (Deutsch/ English/ Chinesisch)
- Nun: Arbeitsplatz in Github erstellen (Done)
- Nun: Aufgabenstellung roh bestimmen
- Nun: Relevante Kenntnisse bereichern (CNN, GCN, Convolution)
- Beim Schreiben der Studienarbeit: Kontaktaufnahme Schreibberatung LUH
https://www.llc.uni-hannover.de/de/schreib-support/individuelle-schreibberatung/ und cc holger.zernetsch@gmail.com 
- Juli/August anmelden —> 4 Monate —> Oktober/November fertig
## Mögliche Aufgabenstellung
- Vorverarbeitung der Bildern
  - Schwellenwertverfahren
  - Rauschunterdrückung
  - Richtungskorrektur
    - Neigungskorrektur
    - Umgekehrt
    - Spiegelbild (nicht normal)
  - Rauschgrenzenentfernung
- Erkennung der Tabellen
  - Erkennung von Texten, Bildern und Tabellen
  - Zerlegung und Wiederherstellung der Tabellenstruktur
- Speicherung
  - Extraktion und Kurzspeichern tabellarischer Daten
  - Organisieren von Daten und Schreiben in die Datenbank
## Relevante Kenntnisse
Python, Elasticsearch, CNN, GCN, Covolution, PCA, SVD
## Wichtige Links von Stand der Technik
- Bild-Convolution: https://blog.csdn.net/chaipp0607/article/details/72236892
- ICDAR: international Conference on Document Analysis and Recognition
- GFTE Graph-based Financial Table Extraction: https://arxiv.org/abs/2003.07560 
Codes davon: https://github.com/Irene323/GFTE
- https://zhuanlan.zhihu.com/p/187112569
- TableNet: Deep Learning model for end-to-end Table detection and Tabular data extraction from Scanned Document Images: https://arxiv.org/abs/2001.01469
- Table Structure Extraction with Bi-directional Gated Recurrent Unit Networks: https://arxiv.org/abs/2001.02501 
Codes davon: https://github.com/saqib22/Table-Structure_Extraction-Bi-directional-GRU?utm_source=wechat_session&utm_medium=social&utm_oi=973919913925124096 
- pro-processing: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.74.50&rep=rep1&type=pdf
- binarization: http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=51158755CB16ED310A63EC033E22C608?doi=10.1.1.61.8&rep=rep1&type=pdf
- https://www.researchgate.net/profile/S-Mandal-3/publication/226626959_Simple_and_effective_table_detection_system_from_document_images/links/56444b4508aef646e6ca792c/Simple-and-effective-table-detection-system-from-document-images.pdf
- Für Schwellenwertverfahren: https://blog.csdn.net/jjddss/article/details/72841141
- Für PCA und SVD: https://blog.csdn.net/weixin_40511249/article/details/121308253 und https://www.jianshu.com/p/1adef2d6dd88 und https://blog.csdn.net/gwplovekimi/article/details/80406808
- einige Arbeiten: https://www.catalyzex.com/s/Table%20Detection
## Mögliche Probleme bei Tabellen
- Einfache Einzelseitentabelle 
(weniger als eine Seite und enthält keine zusammengeführten Zellen)
- Einzelseitentabelle mit zusammengeführten Zellen
- Mehrseitentabelle
- Unvollständige Tabelle
- weiße Linien
- Bunt also BRG-Bild
## Die öffentliche Datasets für die tabellarische Erkennung
- Marmot
- UW3
- UNLV
- ICDAR 2013 / 2019
- PubTabNet
- SciTSR
- TableBank
- FinTab