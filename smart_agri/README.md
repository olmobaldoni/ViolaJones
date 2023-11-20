# Smart Agri

## Train

In `train` sono presenti i dati di training:
- In `pos` sono presenti le 158 immagini positive utilizzate per il training. Da queste immagini sono state ottenute le crops degli insetti
- In `info.dat` sono contenute le annotazioni delle 158 immagini positive.
- In `insects.vec` è il file binario generato da opencv (opencv_createsamples)
    

## Test

In `test` sono presenti i dati di test:
- In `val_data` sono presenti le 18 immagini utilizzate per il test.
- In `annotations` sono presenti le annotazioni delle immagini di test (realizzate con https://www.cvat.ai/). Ho esportato le annotazioni in diversi formati qualora dovessero servire (coco, pascalvoc, yolo, cvat)
- In `risultati` sono presenti i risultati (immagini+detections; file excel) ottenuti con viola jones (5, 15, 20 stadi) ottenuti con le 18 immagini di test.
