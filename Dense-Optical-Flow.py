import cv2
import numpy as np

# Ouverture du flux video
cap = cv2.VideoCapture(

    r"C:\Users\loren\OneDrive - ENSTA Paris\ENSTA\2A\MI204\TP2\TP2_Videos\ZOOM_O_TRAVELLING.m4v")
# Extrait5-Matrix-Helicopter_Scene(280p).m4v")  # Extrait1-Cosmos_Laundromat1(340p).m4v")

ret, frame1 = cap.read()  # Passe à l'image suivante

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Passage en niveaux de gris
# Image nulle de même taille que frame1 (affichage OF)
hsv = np.zeros_like(frame1)
hsv[:, :, 1] = 255  # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

while (ret):
    index += 1
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None,
                                        pyr_scale=0.5,  # Taux de réduction pyramidal
                                        levels=3,  # Nombre de niveaux de la pyramide
                                        # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        winsize=15,
                                        iterations=3,  # Nb d'itérations par niveau
                                        poly_n=7,  # Taille voisinage pour approximation polynomiale
                                        poly_sigma=1.5,  # E-T Gaussienne pour calcul dérivées
                                        flags=0)
    # Conversion cartésien vers polaire
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:, :, 0] = (ang*180)/(2*np.pi)
    hsv[:, :, 2] = (mag*255)/np.amax(mag)  # Valeur <--> Norme

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2, bgr))
    cv2.imshow('Image et Champ de vitesses (Farneback)', result)

    ### CALCUL DES HISTOGRAMMES###
    # Image de base
    yuv_frame1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
    yuv_frame2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    # Extract the chromatic components (u,v)
    U1 = yuv_frame1[:, :, 1]
    V1 = yuv_frame1[:, :, 2]

    U2 = yuv_frame2[:, :, 1]
    V2 = yuv_frame2[:, :, 2]

    # Calculate the 2D histogram of the U and V components
    hist_uv1 = cv2.calcHist([U1, V1], [0, 1], None, [256, 256], [
        0, 256, 0, 256], accumulate=False)
    hist_uv2 = cv2.calcHist([U2, V2], [0, 1], None, [256, 256], [
        0, 256, 0, 256], accumulate=False)

    # Take the logarithm of the histogram to use a log scale
    hist_uv1_log = np.log10(hist_uv1 + 1)
    hist_uv2_log = np.log10(hist_uv2 + 1)

    # Normalize the logarithmic histogram to the range [0, 255]
    cv2.normalize(hist_uv1_log, hist_uv1_log, alpha=0,
                  beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_uv2_log, hist_uv2_log, alpha=0,
                  beta=255, norm_type=cv2.NORM_MINMAX)

    # Convert the logarithmic histogram to a color image using the 'hot' colormap
    colormap_log1 = cv2.applyColorMap(np.uint8(hist_uv1_log), cv2.COLORMAP_JET)
    colormap_log2 = cv2.applyColorMap(np.uint8(hist_uv2_log), cv2.COLORMAP_JET)

    # Apply gaussian filtering on image
    colormap_log1 = cv2.GaussianBlur(colormap_log1, (3, 3), 0)
    colormap_log2 = cv2.GaussianBlur(colormap_log2, (3, 3), 0)

    # Display the histogram
    doubleHisto = np.vstack((colormap_log1, colormap_log2))
    cv2.imshow('Logarithmic Histogram', doubleHisto)

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png' % index, frame2)
        cv2.imwrite('OF_hsv_%04d.png' % index, bgr)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

cap.release()
cv2.destroyAllWindows()

# ##Histogramme
# yuv_frame1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)

# # Extract the chromatic components (u,v)
# u = yuv_frame1[:,:,1]
# v = yuv_frame1[:,:,2]

# # Calculate the 2D histogram of the chromatic components
# hist, x_edges, y_edges = np.histogram2d(u.ravel(), v.ravel(), bins=256)

# # Normalize the histogram
# hist_norm = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# # Display the histogram
# cv2.imshow('2D Histogram', hist_norm)
