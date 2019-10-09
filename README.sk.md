# Anonymizácia tvárí vo videu
Cieľom tohto projektu je detegovať a rozmazať tváre vo videu. Každá tvár vo videu je sledovaná a kontinuálne anonymizovaná. Pre detekciu tváre program využíva Haarcascade klasifikátor a tiež aj Tensorflow API pre rozpoznávanie objektov.

## Príklad anonymizovanej tváre
![Alt text](img/anonymFace.png?raw=true "Anonym face")

## Požiadavky
- Python 3
- OpenCV 4
- Tensorflow

## Modely tváre a tela
V programe sú využívané predtrénované modely pre detekciu tváre a tela. Modely sú trénované na COCO datasete a sú súčasťou knižnice Tensorflow.

- faster_rcnn_inception_v2_coco 
  ( https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models )
- facessed_mobilenet_v2 (**models** priečinok)

## Spustenie skriptu
Skript sa spúšťa nasledujúcim príkazom. Je potrebné zadať cestu k videu.
```
python faceAnonymizer.py -i <videoFile> -s
```

- -i <videoFile> - po argumente i nasleduje cesta k zdrojovému videu. Tento parameter je povinný. 
- -s – argument spúšťa skript v selfie móde. Ak je potrebné rozmazať len tvár v selfie videu, skript dokáže spracovať video rýchlejšie.

## Proces anonymizácie
Anonymizácia tvárí vo videu prebieha v niekoĺkých krokoch. Prvé dva kroky sú inicializačné, ďalšie detegujú a rozmazávajú tváre:

1. Analýza argumentov skriptu - na spustenie skriptu je potrebné zadať adresu videa, ktoré sa má spracovať. Skript sa môže spúšťať aj v režime selfie.
2. Načítanie a inicializácia klasifikátorov Haarcascade a API na detekciu objektov Tensorflow. V tomto kroku sa načíta aj videosúbor.
Po inicializácii sa pre každý obrázok vo videu vykonajú kroky uvedené nižšie:
3. V tejto fáze sa vyhľadajú všetky telá na snímke. Potom sa v oblasti nájdených tiel vyhľadávajú tváre. Ak je tvár nájdená, je anonymizovaná. Ak nie, horná časť tela je anonymizovaná. Tensorflow API na zisťovanie objektov sa používa na detekciu tiel a tvárí. Ak bol skript spustený v režime selfie, celý tento krok sa preskočí a spracovanie je rýchlejšie.
4. Ak sa niektorá tvár nachádza v blízkosti kamery ale telo sa nenašlo, tvár musí byť aj tak anonymizovaná. Z tohto dôvodu detekujeme tvár aj v časti obrazu, kde sa nenachádzajú žiadne telá. V tomto kroku sa používa aj detekcia objektov pomocou Tensorflow.
5. V poslednom kroku sa klasifikátory Haarcascade používajú na detekciu tvárí, uší a očí. Vďaka týmto klasifikátorom môžeme detegovať tvár z profilu alebo tvár, ktorá je čiastočne zakrytá.
Po týchto krokoch sú všetci ľudia vo videu anonymizovaní, bez ohľadu na to, či sú ich tváre blízko fotoaparátu alebo ďaleko. Dokonca aj zakryté tváre sú rozmazané.
