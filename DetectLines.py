## Import modules
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ExtractLines(ImageBytes,MinLineHeight=8,MinGap=1,Padding=6,RowInkThreshold=None):
    ## Convert the ImageBytes Object to np array
    ImageArray1D=np.frombuffer(ImageBytes,np.uint8)
    ## Convert the array to Image
    ImageArray = cv2.imdecode(ImageArray1D,cv2.IMREAD_UNCHANGED)
    if ImageArray is None:
        return None
    ## Check if Image is already in grayscale
    if len(ImageArray.shape)==2:
        GrayImage=ImageArray
    else:
        if len(ImageArray.shape)==4:
        ## Convert the image to grayscale
            GrayImage=cv2.cvtColor(ImageArray,cv2.COLOR_BGRA2GRAY)
        else:
            GrayImage=cv2.cvtColor(ImageArray,cv2.COLOR_BGR2GRAY)

    ## Use reverse Otsu's Thresholding to convert the GrayScale image to Binary Image
    BestThreshold,BinaryImage=cv2.threshold(GrayImage,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ## Create a kernel of size 15x1 (Horizontal kernel)
    Kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,1))
    ## Apply closing operation to bridge the gaps between words using the kernel

    ClosedBinary=cv2.morphologyEx(BinaryImage,cv2.MORPH_CLOSE,Kernel)
    ## Get the number of pixels with value = 1 (row wise)
    NPixels=np.sum(ClosedBinary,axis=1).astype(np.float64)
    ## NPixels dimension is (Height x 1)
    ## Compute the overall Ink Threshold
    if RowInkThreshold is None:
        RowInkThreshold=0.1 * NPixels.max() if NPixels.max()>0 else 0
    ## Map rows that have number of pixels greater than Threshold to 1 else 0
    IsTextLine=NPixels>RowInkThreshold
    ## Create an empty list to collect the lines
    Lines=list()

    start=None
    for LineNumber, Flag in enumerate(IsTextLine):
        if Flag and start is None: ## Select the Line coordinate that has text
            start=LineNumber
        elif not Flag and start is not None: ## Detects end coordinate (without text)
            Lines.append((start,LineNumber-1))
            start=None ## Set None to from the next Lines
    if start is not None: ## Safety check if text expands till the end of image
        Lines.append((start,len(IsTextLine)-1))

    MergedLines=list()
    ## Merge each line with a small gap / padding
    for Line in Lines:
        if MergedLines and Line[0]-MergedLines[-1][1]<=MinGap:
            MergedLines[-1]=(MergedLines[-1][0],Line[1])
        else:
            MergedLines.append(list(Line))

    ## Drop the short bands (false bands) i.e Add lines if the gap between them is larger than Minimum Line Height
    FinalLines=[Line for Line in MergedLines if (Line[1]-Line[0]) >= MinLineHeight]

    ## Crop each line from the uploaded image with padding and add to buffer
    height, width=GrayImage.shape[:2]
    Buffer=[]
    for ycord1,ycord2 in FinalLines:
        yCord1=max(0,ycord1-Padding)
        yCord2=min(height,ycord2+Padding)
        Cropped=GrayImage[yCord1:yCord2,0:width]
        ## Convert the cropped image to png and add to buffer
        WasSuccessful,EncodedImage=cv2.imencode(".png",Cropped)
        if WasSuccessful == True:
            Buffer.append(EncodedImage.tobytes())
        # print(f"Detected {len(Lines)} lines")
    return Buffer