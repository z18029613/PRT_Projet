# -*- coding: utf-8 -*-
"""PRT_Vitesse.ipynb
Test codes pour PRT 19/01/2023
Lorenzo ROSADO
"""

#Libraries importées
import numpy as np
from numpy.fft import fft,ifft, fftshift
import matplotlib.pyplot as plt
import scipy as sc
import scipy.io.wavfile as wav



#Bloc d'importation du fichier 
uploaded = "C:/Users/easyr/Desktop/Son_a.wav"
#Variable contenant le son : 
    
Son_AAA = wav.read(uploaded)

Son_freq = Son_AAA[0]
Son_array = Son_AAA[1][:,0] # <!> il serait de taille N x 2 car il y a deux canaux audios, oreille gauche et oreille droite.
                            # ici nous n'en utilisons qu'une seule. A l'avenir son mono serait mieux

Son_freq = int(Son_freq/2) # fe/2
fe = Son_freq
Son_fft = fft(Son_array) / 2
Son_comprime = np.abs ( Son_fft )

#écrire le fichier wav et le télécharger avec méthode 1
test = wav.write("test.wav", Son_freq, Son_comprime )
#Avec cette méthode le son sort presque détruit, on ne reconnait même pas le signal d'entrée

Son_AAA_fft = np.fft.fft(Son_array, norm="ortho")
Son_AAA_selfconvol  = np.abs(np.fft.ifft(np.abs(Son_AAA_fft), norm="ortho"))

Son_AAA_selfconvol=Son_AAA_selfconvol/Son_AAA_selfconvol[0]


N = int(len(Son_array /2))
x = np.linspace(-fe,fe,N)
plt.plot(x,np.abs(np.fft.fftshift(np.fft.fft(Son_array))) , 'b',label = "son de base")
plt.plot(x,np.abs(np.fft.fftshift(np.fft.fft(Son_comprime))) ,'r',label = "son de comprimé")
plt.legend()
plt.title("Fréquence d'un son produit par la prononciation d'un A et sa compression.")
plt.xlabel("frequence")
plt.ylabel("log (|S(f)|")
plt.show()
