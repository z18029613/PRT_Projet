# -*- coding: utf-8 -*-
"""PRT_Vitesse.ipynb
Test codes pour PRT 19/01/2023
Lorenzo ROSADO
"""

"""Libraries importées"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.fft import fft , ifft, fftshift
 
plt.close(fig='all')

"""CONSTANTES"""
DISPLAY =  0 #True

"""Bloc d'importation du fichier """
uploaded = "Son_a.wav"

"""Variable contenant le son : """
Son_AAA = wav.read(uploaded)
Son_freq = Son_AAA[0]           # fréquence du fichier
fe = Son_freq
Son_array = Son_AAA[1][:,1]          # Le son ici est stereo donc on ne selectionne qu'un des deux canaux.


##############################################################################################################################################
"""Méthode 1 : Compression"""
#Son_array_destruct=Son_array/2
#écrire le fichier wav et le télécharger avec méthode 1
#file_test = wav.write("test_compression.wav", Son_freq*2, Son_array_destruct )
#Avec cette méthode le son sort presque détruit, on ne reconnait même pas le signal d'entrée
#plt.plot(x,np.abs(np.fft.fftshift(np.fft.fft(Son_array_destruct[:,0]))) ,'r',label="TF du son après compression")
##############################################################################################################################################

"""Méthode 2 : Passage par zéro, pseudo période"""
#Pour filtrer, nous avons besoin de la TF du signal pour faire la convolution avec le filtre
Son_fft = fft(Son_array)

N = int(len(Son_fft)/2)  
x = np.arange( -N , N+1 ) #Axe des abscisses pour la TF shift

"""Filtre Passe Bas Porte"""
porte           = np.ones(Son_array.shape)
i_porte         = 35000
porte[  N - i_porte : N + i_porte ]   =  0
Son_filtre_fft  = Son_fft * porte 

Son_filtre = np.real ( ifft(Son_filtre_fft) )   # Problème de décalage sur la TF, post ifft nous avons toujours des complexes

tmp = np.zeros (Son_AAA[1].shape )              # ¨Remplissage des deux canaux audios
for i in range (Son_AAA[1].shape[0]) : 
    tmp [i,0] = Son_filtre[i]
    tmp [i,1] = Son_filtre[i]
tmp = np.array( tmp , dtype = "int16")          # Convertion dans le bon format  pour l'enregistrement

file_sonpostfiltre = wav.write("Son_post_Porte.wav", Son_freq , tmp )

"""Detection des passages par Zero"""

Passage_Zero = np.array([(0,0)])
for i in range (Son_filtre.shape[0] -1) : 
    if np.sign(Son_filtre[i]) != np.sign(Son_filtre[ i+1 ]) : 
        A = np.array([ (i ,Son_filtre[i] ) ])
        Passage_Zero = np.append (Passage_Zero ,  A  , axis= 0 )

##############################################################################################################################################

if (DISPLAY) : 
    plt.figure(1)
    plt.plot(x,  np.abs( fftshift( Son_fft           )) , 'b',label="TF du son 'a' ")
    plt.plot(x , np.abs( fftshift( Son_filtre_fft    )) , 'r--',label="TF filtré par porte")
    plt.xlabel("frequence")
    plt.ylabel("|S(f)|")
    plt.title("TF des signaux")
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.plot(x, fftshift(porte ))
    plt.title("Filtre porte passe bas")
    plt.show()
    
    plt.figure(3)
    plt.title("Signaux dans l'espace temporel")
    plt.plot(Son_array , 'b' , label = "Signal initial")
    plt.plot(Son_filtre, 'r--' , label = "Signal filtre")
    plt.legend()   
    plt.show()
    
##############################################################################################################################################
# """Construction du filtre passe bas"""
# n = np.arange(0,len(Son_AAA_fft))
# n_temp = np.zeros ( np.shape(Son_array) , dtype="int16")
# n_temp[:,0],n_temp[:,1] = n , n 
# n = n_temp

# fe = Son_freq # sampling frequency

# Freq = np.linspace(-fe/2, fe/2, len(n))
# F_temp = np.zeros ( np.shape(Son_array) , dtype="int16")
# F_temp[:,0],F_temp[:,1] = Freq, Freq   #Ici ce Freq correspond a tous les points de l'abscisse
# Freq = F_temp

# h = np.zeros ( np.shape(Son_array))# , dtype="int16") #
# H = np.zeros ( np.shape(Son_array))

# freq_coupure = 100
# std = freq_coupure
# coef = 100

# h[:,0] = 1/np.sqrt(2 * np.pi * std**2)  * np.exp(- Freq[:,0]**2 / (2* std**2) ) * coef
# h[:,1] = 1/np.sqrt(2 * np.pi * std**2)  * np.exp(- Freq[:,1]**2 / (2* std**2) )  *coef

# # H[:,0] = np.fft.fft(h[:,0])  
# # H[:,1] = np.fft.fft(h[:,1]) 
# H[:,0] = h[:,0]
# H[:,1] = h[:,1]
# Son_filtre = H * Son_AAA_fft 

# plt.plot(x , np.abs(np.fft.fftshift(np.fft.fft(Son_filtre[:,0]))) , 'r',label="TF filtré par H")
# plt.legend()
# plt.title("Fourier Transform")
# plt.figure(1)
# plt.show()

# plt.figure(2)
# plt.plot(Freq[:,0] ,  np.abs (    np.fft.fftshift(H) ) , label = "|H|")
# plt.title("H Low pass filter")
# plt.legend()
# plt.show()

# Son_AAA_postfiltre = np.array(  np.real( np.fft.ifft (Son_filtre) ) , dtype = "int16")

# plt.figure(3)
# plt.plot(Son_AAA_postfiltre,label="son post filtre")
# plt.title("Filtered sound")
# plt.legend()
# plt.show()

# file_sonpostfiltre = wav.write("sonpostfiltre.wav", Son_freq , Son_AAA_postfiltre )






