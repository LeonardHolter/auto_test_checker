import cv2 
import numpy as np
#Importerer noen funksjoner fra en Inder
import OCR_Verktøy

#######################

path = "/Users/leonard/Documents/Python/OCR/Test Prøve.png"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
svar = [1,2,0,1,4]

#######################


img = cv2.imread(path)  #Laster bildet inn i memory

######### Preprocessing #########
img = cv2.resize(img, (widthImg, heightImg)) #Zoomer litt inn
imgCountors = img.copy()
imgBiggestContors = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Fjerner fargene
ImgBlur = cv2.GaussianBlur(imgGray,(5,5),1) #Gjør finere detaljer uklare
ImgCanny = cv2.Canny(ImgBlur, 20, 50) #Lokaliserer kantene





contours, hierarchy = cv2.findContours(ImgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #Lokaliserer ytterlinjene
cv2.drawContours(imgCountors, contours, -1, (0, 255, 0), 10)  #Tegner en grønn strek over ytterlinjene



######### Finner rektangler #########
rectCon = OCR_Verktøy.rectContour(contours)
biggestContour = OCR_Verktøy.getCornerPoints(rectCon[0])





if biggestContour.size != 0:
    cv2.drawContours(imgBiggestContors, biggestContour,-1, (0,255,0), 10)

    biggestContour = OCR_Verktøy.reorder(biggestContour)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthImg, 0],[0, heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg,heightImg))





    #APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]



   #Deler hvert svaralternativ som et eget bilde i en liste
    boxes = OCR_Verktøy.splitBoxes(imgThresh)
   




    #Skaffer verdier på hvor mye hvitt det er i hvert box-bilde
    myPixelVal = np.zeros((questions, choices))
    countC = 0
    CountR = 0



    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[CountR][countC] = totalPixels
        countC += 1
        if (countC == choices):
            CountR +=1 
            countC = 0


    #Jeg finner rikig indeks verdi til svaret med mest hvitt i samme kolonne
    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr==np.amax(arr))
        myIndex.append(myIndexVal[0][0])





    # RETTING
    grading = []
    for x in range(0, questions):
        if svar[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)



######### KARAKTER #########
    score = sum(grading)/questions * 100 
    
    if score > 93: 
        print("Karakteren din blir 6")
    elif score > 70 and score < 93:
        print("Karakteren din blir 5")
    elif score > 40 and score < 70:
        print("Karakteren din blir 4")
    elif score > 31 and score < 40:
        print("Karakteren din blir 3")
    elif score > 18 and score < 31:
        print("Karakteren din blir 2")
    else:
        print("Karakteren din blir 1")


cv2.imshow("box", boxes[2])

imgBlank = np.zeros_like(img) #Lager et sort bilde med samme form som img

imageArray = ([img, imgGray, ImgBlur, ImgCanny],
              [imgCountors,imgBiggestContors,imgWarpColored,imgThresh]) #Lager en liste med alle bilde-objektene
imgStacked = OCR_Verktøy.stackImages(imageArray, 0.5) #Sender lista inn i galleri funksjonen


cv2.imshow("Stacked Images", imgStacked)  #Viser galleriet
cv2.waitKey(0) #Setter ingen forsinkelse