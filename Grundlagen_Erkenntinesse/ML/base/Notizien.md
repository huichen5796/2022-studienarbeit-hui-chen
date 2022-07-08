# Ⅰ. Logistic Regression
![logistic_regression1](../Bilder/logistic_regression1.png)

1. Unsere Aufgabe ist es, eine Gruppe von W, b zu finden, so dass unser Modell y' = σ(W^Tx+b) y bei gegebenem x korrekt vorhersagen kann. Hier können wir denken, dass, solange das berechnete y' größer als 0,5 ist, y' näher an 1 liegt, sodass es als "ist eine Katze" vorhergesagt werden kann, andernfalls ist es "keine Katze".
2. Wie zu suchen

    - Wir können eine Verlustfunktion definieren, um die Differenz zwischen y' und y zu messen:

        L(y',y) = -[y·log(y')+(1-y)·log(1-y')]
3. Abstrakt
    - alle Input können in einen Zeilenvektor geschrieben werden:

            X = (x(1)，x(2)，...，x(m))

        x(i) stellt manchmal ein Beispielbild dar

    - ähnlich für alle wahre Etikette: 
    
            Y = (y(1)，y(2)，...，y(m))

    - alle Prognosen: 
       
            Y' = (y'(1)，y'(2)，...，y'(m))

    - Cost function:

        stellt den durchschnittlichen Verlust über alle Trainingsgebiete dar

        J(W,b) = 1/m·Σmi=1L(y'(i),y(i))

    **Daher lässt sich unsere Lernaufgabe in einem Satz ausdrücken: Find W,b that minimize J(W,b)**

4. Gradient Decent

    ![GD](../Bilder/InkedGD_LI.jpg)
        
        w := w - α(dJ/dw)

    α ist learning rate

5. Forward und Backward Propagation

    ![F_B](../Bilder/F_B_propagation.png)

## abschließend:
- Logistisches Regressionsmodell: y' = σ(WTx+b), denken Sie daran, dass die verwendete Aktivierungsfunktion die Sigmoidfunktion ist.
- Verlustfunktion: L(y',y) = -[y·log(y')+(1-y)·log(1-y')], um die Differenz zwischen dem vorhergesagten Wert y' und dem wahren Wert y zu messen , je kleiner desto besser ist es gut.
- Kostenfunktion: Verlustmittelwert, J(W,b) = 1/m·Σmi=1L(y'(i),y(i)), ist eine Funktion von W und b, und der Lernprozess besteht darin, W und zu finden b, so dass J (W,b) Minimierungsprozess. Um den Mindestwert zu finden, verwenden Sie den Gradientenabstieg.

Schritte zum Trainieren des Modells:
- W und b initialisieren
- Geben Sie die Lernrate und die Anzahl der Iterationen an
- Bei jeder Iteration wird der entsprechende Gradient (die partielle Ableitung von J nach W, b) basierend auf dem aktuellen W und b berechnet, und dann werden W und b aktualisiert
- Bekommen Sie am Ende der Iteration W und b, bringen Sie es zur Vorhersage in das Modell und testen Sie die Genauigkeit auf dem Trainingssatz bzw. dem Testsatz, um das Modell zu bewerten

![helper_functions](../Bilder/helper_functions.png)


# Ⅱ. Shallow Neural Network und Deep Neural Network

Das neuronale Netzwerk ist es, eine oder mehrere versteckte Schichten auf der Grundlage der logistischen Regression zu hinzufügen. Das Folgende ist das einfachste neuronale Netzwerk mit nur zwei Schichten:

![snn](../Bilder/snn.png)

- Notation
    - z = wx+b
    - a = σ(z)
    - Die Indizes 1, 2, 3, 4 repräsentieren das i-te Neuron (Einheit) der Schicht 
    - Die hochgestellten Zeichen [1], [2] usw. repräsentieren den aktuellen Schicht
    - Die x1, x2, x3, x4 im obigen Bild repräsentieren nicht 4 Samples! Aber vier Merkmale (Werte in 4 Dimensionen) einer Probe! Wenn Sie m Proben haben, bedeutet dies, den obigen Vorgang m-mal zu wiederholen.
- Deep Neural Network

    - ist also Shallow Neural Network mit vielen Schichten
    - Beachten Sie, dass wir im tiefen neuronalen Netzwerk die Aktivierungsfunktion „ReLU“ in der mittleren Schicht anstelle der Sigmoid-Funktion verwenden und die Sigmoid-Funktion nur in der letzten Ausgabeschicht verwenden, da die ReLU-Funktion beim Auffinden von Gradienten schneller ist. Es kann auch das Phänomen des verschwindenden Gradienten bis zu einem gewissen Grad verhindern, weshalb es häufig in tiefen Netzwerken verwendet wird.
# Ⅲ. Convolutional Neural Network
### ähnlich wie Logistic Regression
![CNN1](../Bilder/CNN1.png)

1. input ---> X, shape = (8, 8, 3)
2. W1, shape = (3, 3, 3, 4), hier bedeutet 4 4 filter.
3. Z1, shape = (6, 6, 4), also no padding
4. activierende Funktion, z.B. ReLU, nach Aktivieren wird Z1 A1 sein, shape = (6, 6, 4)

![CNN2](../Bilder/CNN2.png)
### Begriff
1. convolution
2. pooling
3. padding
### Struktur von CNN
1. Convolutional layer - CONY
    = filters + activate function
    - hyper parameters: 
        - amount of filters
        - size of filters
        - stride
        - form of padding
            - valid
            - same
       - which activate function
2. Pooling layer - POOL
    - Maxpooling
    - Averagepooling
    - size of block ---> (2, 2)
    - stride ---> 2
3. Fully Connected layer - FC
    - ReLU
    - Softmax
4. Dropout
    dazu: https://blog.csdn.net/qq_52302919/article/details/122796577
    
    ![FC](../Bilder/FC.png)
    Die letzten beiden Spalten kleiner Kugeln stellen zwei vollständig verbundene Schichten dar. Nachdem die letzte Faltungsschicht abgeschlossen ist, wird das letzte Pooling durchgeführt, 20 12*12-Bilder werden ausgegeben, und dann wird eine vollständig verbundene Schicht zu einem 1*100-Vektor.

    Tatsächlich gibt es 20*100 12*12 Faltungskerne, die gefaltet werden. Für jedes Eingangsbild wird eine Kernfaltung derselben Größe wie das Bild verwendet,so dass das gesamte Bild zu einer Zahl wird.
### CNN vs. NN

Tatsächlich unterscheidet sich CNN nicht sehr von dem zuvor erlernten neuronalen Netzwerk. Das herkömmliche neuronale Netzwerk ist eigentlich ein Stapel aus mehreren FC-Schichten. CNN ist nichts anderes als die Änderung von FC in CONV und POOL, wodurch die traditionelle Schicht aus Neuronen in eine Schicht aus Filtern umgewandelt wird.

# Ⅳ Transfer Learning
- Verwenden werden die Modelle, die von anderen für das Verständnis von Bildinhalten trainiert wurden.
- Ersetzen wird die letzte Schicht
- Nur die letzte Schicht muss trainiert werden
- Kann aber auch alle Parameter trainiert werden - Pretraining

Durch die Verwendung von Gewichtungen, die von anderen vortrainiert wurden, ist es möglich, auch mit einem kleinen Datensatz eine gute Leistung zu erzielen.
https://nanonets.com/documentation/
https://blog.csdn.net/john_bh/article/details/106231354?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165470187716782246479348%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=165470187716782246479348&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-15-106231354-null-null.article_score_rank_blog&utm_term=Pytorch&spm=1018.2226.3001.4450


# Ⅴ Classification, Detection, Segmentation

### Classification
- That is, the image is structured into a certain category of information, and the picture is described by a predetermined category (string) or instance ID. This task is the simplest and most basic image understanding task, and it is also the first task for deep learning models to achieve breakthroughs and large-scale applications. Among them, ImageNet is the most authoritative evaluation set, and the annual ILSVRC has spawned a large number of excellent deep network structures, providing a foundation for other tasks. In the application field, face, scene recognition, etc. can be classified as classification tasks.
- Simply put ---> determine what category the entire image belongs to

### Detection
- The classification task is concerned with the whole, and the content description of the whole picture is given, while the detection focuses on a specific object target, and requires the category information and location information of the target to be obtained at the same time. Compared with classification, detection gives an understanding of the foreground and background of the picture. We need to separate the target of interest from the background and determine the description (category and position) of this target. Therefore, the output of the detection model is a List, each item of the list uses a data set to give the category and location of the detected target (commonly represented by the coordinates of the rectangular detection frame).

### Segmentation
- Segmentation includes semantic segmentation (semantic segmentation) and instance segmentation (instance segmentation). The former is an extension of the front-background separation, which requires the separation of image parts with different semantics, while the latter is an extension of the detection task, which requires describing the outline of the target ( It is finer than the detection frame). Segmentation is a pixel-level description of an image, which gives each pixel category (instance) meaning, and is suitable for scenarios with high understanding requirements, such as the segmentation of roads and non-roads in autonomous driving.

# Ⅵ Pytorch

### for use see beispiel_pytorch.py
- Guide für Pztorch ---> https://pytorch.org/tutorials/beginner/basics/intro.html
- install ---> https://pytorch.org/get-started/locally/
- The `torchvision` package consists of popular datasets, model architectures, and common image transformations for computer vision. ---> https://pytorch.org/vision/stable/index.html
- Datasets von Pytorch

# Ⅶ Modelle für Detection

### 2-stage modelle(not end to end)
- R-CNN
- FAST R-CNN
- FASTER R-CNN
- MASK R-CNN

detectron2 --- https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py

### 1-stage modelle(end to end)
- SSD
- TOLO
  - YOLO v1
  - YOLO v2
  - YOLO v3
  - YOLO v4
  - YOLO v5 --- https://github.com/ultralytics/yolov5


# Ⅶ Datasets von Tabellen
- https://github.com/doc-analysis/TableBank
- https://github.com/ibm-aur-nlp/PubTabNet
- https://github.com/Academic-Hammer/SciTSR
- Download processed Marmot dataset: https://drive.google.com/file/d/1irIm19B58-o92IbD9b5qd6k3F31pqp1o/view?usp=sharing

# Ⅷ code implementations (in TensorFlow and PyTorch)
https://github.com/asagar60/TableNet-pytorch. Hier werden die Datasets genutzt.

# Ⅸ Single Object Segmentation
https://www.jianshu.com/p/7086ded792b2