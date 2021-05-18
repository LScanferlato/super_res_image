# -*- coding: utf-8 -*-
import cv2

from cv2 import dnn_superres

def main():
    input_path = "./video/in.mp4"
    output_path = "./video/out.mp4"
    # edsr, espcn, fsrcnn or lapsrn
    algorithm = "lapsrn"
    # Rapporto di ingrandimento, 2, 3, 4, 8, selezionare in base alla struttura del modello 
    scale = 2
    # Percorso del modello 
    path = "./model/LapSRN_x2.pb"

    # Apri il video 
    input_video = cv2.VideoCapture(input_path)
    # Dimensioni di codifica dell'immagine in ingresso 

    ex = int(input_video.get(cv2.CAP_PROP_FOURCC))

    # Ottieni la dimensione dell'immagine video in uscita
    # Se il video non è aperto 
    if input_video is None:
        print("Could not open the video.")
        return

    S = (
    int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale, int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale)

    output_video = cv2.VideoWriter(output_path, ex, input_video.get(cv2.CAP_PROP_FPS), S, True)

    # Leggi il modello del superingrandimento 
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(path)
    sr.setModel(algorithm, scale)

    while True:
        ret, frame = input_video.read()  # Cattura un fotogramma dell'immagine 
        if not ret:
            print("read video error")
            return
        # Immagine sovracampionata 
        output_frame = sr.upsample(frame)
        output_video.write(output_frame)

        cv2.namedWindow("Upsampled video", cv2.WINDOW_AUTOSIZE);
        cv2.imshow("Upsampled video", output_frame)

        cv2.namedWindow("Original video", cv2.WINDOW_AUTOSIZE);
        cv2.imshow("Original video", frame)

        c = cv2.waitKey(1);
        # esc
        if 27 == c:
            break

    input_video.release()
    output_video.release()


if __name__ == '__main__':
    main()
