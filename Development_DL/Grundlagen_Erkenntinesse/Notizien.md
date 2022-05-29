# Ⅰ. Logistic Regression - ein einfachstes neuronale Netzwerk
![logistic_regression1](Development-DL\\Bilder\\logistic_regression1.png)

1. Unsere Aufgabe ist es, eine Gruppe von W, b zu finden, so dass unser Modell y' = σ(WTx+b) y bei gegebenem x korrekt vorhersagen kann. Hier können wir denken, dass, solange das berechnete y' größer als 0,5 ist, y' näher an 1 liegt, sodass es als "ist eine Katze" vorhergesagt werden kann, andernfalls ist es "keine Katze".
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

![GD](./Development-DL/Bilder/InkedGD_LI.jpg)



