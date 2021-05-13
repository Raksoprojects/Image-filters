from PIL import Image
import requests
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np

#Getting image from the internet
response = requests.get("https://media.comicbook.com/2020/08/cyberpunk-2077-1--1233341-1280x0.jpeg")
img = Image.open(BytesIO(response.content))
Image._show(img)
data = np.asarray_chkfinite(img) #changing image to array
histogram = np.zeros((768)) #creating an array which will be our histogram

#looping through an image to make a histogram
for i in data[:,:,0]:
    for j in i:
        histogram[j] += 1
for i in data[:,:,1]:
    for j in i:
        histogram[j+256] += 1
for i in data[:,:,2]:
    for j in i:
        histogram[j+512] += 1

x = np.linspace(0,768,768)

plt.figure(1)
plt.title("RGB Histogram")
plt.xlabel("Color's intensities")
plt.ylabel("Value")
plt.bar(x,histogram) #plotting colors histogram

#Creating grayscale image and histogram of it
greyimg = np.mean(data, axis=2) #uśrednianie wartości red, green i blue
grey_histogram = np.zeros((256))
y = np.linspace(0,256,256)
for i in greyimg:
    for j in i:
        grey_histogram[int(round(j))] += 1 #rzutuję na inta zaokrąglone wartości(float), pomocne przy tworzeniu histogramu

image_in_grey = Image.fromarray(greyimg) #creating a gray image from array

plt.figure(2)
plt.title("Grayscale Histogram")
plt.xlabel("Gray intensity")
plt.ylabel("value")
plt.bar(y,grey_histogram)

#Filtr Sobela
Gx_d = np.array([[1,0,-1], [2,0,-2], [1,0,-1]]) #tworzenie dwóch masek
Gy_d = np.array([[1,2,1], [0,0,0], [-1,-2,-1]]) #y oraz x
Gx = np.convolve(np.ndarray.flatten(greyimg), np.ndarray.flatten(Gx_d), mode ="same") #aplikowanie konwolusji na obydwu zmaskowanych wynikach
Gy = np.convolve(np.ndarray.flatten(greyimg), np.ndarray.flatten(Gy_d), mode = "same")

Gx = Gx.astype(int) #wymuszanie typu zmiennych 
Gy = Gy.astype(int)

G = np.sqrt(np.add(np.square(Gx), np.square(Gy) ) ) #obliczanie wspólnego obrazka filtrów x i y
G = np.reshape(G,(data.shape[0], data.shape[1])) #przywracanie mu kształtu obrazka po uwczesnym spłaszczeniu
G = G / G.max() * 255 # normalizacja 

sobel = Image.fromarray(G) #tworzenie obrazka sobela

#Filtr Roberts'a
Rx_d = np.array([[1,0], [0,-1]]) #tworzenie dwóch masek
Ry_d = np.array([[0, 1], [-1, 0]]) #y oraz x
Rx = np.convolve(np.ndarray.flatten(greyimg), np.ndarray.flatten(Rx_d), mode ="same") #aplikowanie konwolusji na obydwu zmaskowanych wynikach
Ry = np.convolve(np.ndarray.flatten(greyimg), np.ndarray.flatten(Ry_d), mode ="same")

Rx = Rx.astype(int) #wymuszanie typu zmiennych
Ry = Ry.astype(int)

R = np.sqrt(np.add(np.square(Rx), np.square(Ry) ) ) #obliczanie wspólnego obrazka filtrów x i y
R = np.reshape(R,(data.shape[0], data.shape[1])) #przywracanie mu kształtu obrazka po uwczesnym spłaszczeniu
R = R / R.max() * 255 # normalizacja 

roberts = Image.fromarray(R)

#Filtr Canny'ego
H= np.array([[2,4,5,4,2], [4,9,12,9,4], [5,12,15,12,5], [4,9,12,9,4], [2,4,5,4,2]]) #utworzenie maski filtru Gaussa, 5x5 
H = np.multiply((1/159), H)

gaussed_image = np.convolve(np.ndarray.flatten(greyimg), np.ndarray.flatten(H), mode ="same") #aplikowanie konwolusji
gaussed_image = np.reshape(gaussed_image,(data.shape[0], data.shape[1]))

#Używamy filtru sobela Gx i Gy

Gx_gaussed = np.convolve(np.ndarray.flatten(gaussed_image), np.ndarray.flatten(Gx_d), mode ="same") #aplikowanie konwolusji na obydwu zmaskowanych wynikach
Gy_gaussed = np.convolve(np.ndarray.flatten(gaussed_image), np.ndarray.flatten(Gy_d), mode = "same") #podobnie jak w Sobelu, natomiast na "zgaussowanym" obrazku

Gx_gaussed= Gx_gaussed.astype(int) #assertowanie typu
Gy_gaussed = Gy_gaussed.astype(int)
phi = np.arctan2(Gx_gaussed,Gy_gaussed) # liczenie kątów krawędzi
G_gaussed = np.sqrt(np.add(np.square(Gx), np.square(Gy) ) ) # obliczanie filtru sobela dla zmienionego obrazka
G_gaussed = np.reshape(G_gaussed,(data.shape[0], data.shape[1])) # ustawianie poprawnyego wymiaru
G_gaussed = G_gaussed / G_gaussed.max() * 255 #normalizacja
phi = np.reshape(phi, (data.shape[0], data.shape[1])) # ustawianie poprawnego wymiaru

#Implementowanie non-max-supresion

x1, x2 = G_gaussed.shape  #pobieranie wymiarów obrazka
maxi = np.zeros((x1,x2)) #utworzenie zerowego arraya
maxi = maxi.astype(int) 
kat = phi * 180. / np.pi 
kat[kat<0] += 180 #wymuszenie wartości kątów od 0 do 180 stopni

#pętle tworzące nową tablicę zawierającą "wychudzone" krawędzie
#posiłkowane artykułami z internetu, sprawdzanie algorytmu
for i in range(1,x1-1):
    for j in range(1,x2-1):
        q = 255
        r = 255
        if (0 <= kat[i,j] < 22.5) or (157.5 <= kat[i,j] <= 180):
            q = G_gaussed[i, j+1]
            r = G_gaussed[i, j-1]
        elif (22.5 <= kat[i,j] < 67.5):
            q = G_gaussed[i+1, j-1]
            r = G_gaussed[i-1, j+1]
        elif (67.5 <= kat[i,j] < 112.5):
            q = G_gaussed[i+1, j]
            r = G_gaussed[i-1, j]
        elif (112.5 <= kat[i,j] < 157.5):
            q = G_gaussed[i-1, j-1]
            r = G_gaussed[i+1, j+1]

        if(G_gaussed[i,j] >= q) and (G_gaussed[i,j] >=r):
            maxi[i,j] = G_gaussed[i,j]
        else:
            maxi[i,j] = 0 

canny = Image.fromarray(maxi)

#Wyświetlanie poszczególnych filtrów i szarego obrazka dla porównania
plt.figure(3)
plt.title("Szary obrazek")
plt.imshow(image_in_grey)

plt.figure(4)
plt.title("Filtr Sobel'a")
plt.imshow(sobel)

plt.figure(5)
plt.title("Filtr Robertsa")
plt.imshow(roberts)

plt.figure(6)
plt.title("Filtr Canny'ego")
plt.imshow(canny)

plt.show()