import sys
import getopt
import math
import numpy as np
import cv2

from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage import img_as_float # Using an image from OpenCV with skimage
from skimage import img_as_ubyte # Using an image from skimage with OpenCV
from skimage import color
from skimage import io

import skimage.filters
from skimage.filters.rank import enhance_contrast
from skimage.filters.rank import median

from skimage.morphology import disk
from skimage.morphology import reconstruction
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate

### Constants
FILENAME = 'image.jpg'
# TODO um / pixel microMeterPerPixel = 20.0 # 19.8
MICRO_METER_PER_PIXEL = 17.39

global DEBUG_FILTERING
global DEBUG_SEGMENTING
global DEBUG_EDITING
global DEBUG_PLOT

HYBRID_FILTER = True

global imageOriginal
global imagePreProcessed
global imageFiltered
global imageSegmented
global imageEdited
global labels
global distance

global listAreas, listFittedEllipse, listEquivalentCircularAreaDiameter, listLBCRadius

global listDebugImages, listDebugTitles

listDebugImages = list()
listDebugTitles = list()

def analyze():
    global imageOriginal, imagePreProcessed, imageFiltered, imageSegmented, imageEdited
    global listAreas, listFittedEllipse, listEquivalentCircularAreaDiameter, listLBCRadius
 
    ### Load image ###
    imageOriginal = loadImage()

    ### Pre-process image ###
    imagePreProcessed = preProcessImage(imageOriginal)
    
    ### Filtering ###
    imageFiltered = filterImage(imagePreProcessed)

    ### Segmenting ###
    imageSegmented = segmentImage(imageFiltered)

    ### Editing ###
    imageEdited = editImage(imageSegmented)

    ### Measure ###
    # TODO
    label_img = ndi.label(imageEdited)[0] #label(imageEdited)
    #local_maxi = peak_local_max(imageEdited, indices=False, footprint=np.ones((3, 3)), labels=label_img, num_peaks_per_label=1)
    #print("Local Maxi: " + str(local_maxi) + ")")
    regions = regionprops(label_img)

    listGrainImages = list()
    listHeights = list()
    listWidths = list()
    listAreas = list()
    listCPM = list()
    listECPD = list()

    listPerimeters = list()
    listCentroids = list()
    listEquivalentCircularAreaDiameter = list()
    listLBCRadius = list()

    listFiberLength = list()
    listFiberWidth = list()
    listFittedEllipse = list()
    listMajorAxisLength = list()
    listMinorAxisLength = list()
    listIsConvex = list()

    fig, ax = plt.subplots()
    ax.imshow(imageEdited, cmap=plt.cm.gray)
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=1.0)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.0)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=1.0)

        boundingBoxRowMin, boundingBoxColMin, boundingBoxRowMax, boundingBoxColMax = props.bbox

        ### Measurements ###
        cv2_image = img_as_ubyte(props.image)
        _, contours, hierarchy = cv2.findContours(cv2_image, 1, 2)
        contour = contours[0]

        listGrainImages.append(props.image)

        # Area
        listAreas.append(props.area)

        # Horizontal Feret’s Diameter (HFD): The distance between two parallel lines at horizontal direction that do not intersect the image.
        # Width of boundingbox
        listWidths.append(boundingBoxRowMax - boundingBoxRowMin)

        # Vertical Feret’s Diameter (VFD): The distance between two parallel lines at vertical direction that do not intersect the image.
        # Height of boundingbox
        listHeights.append(boundingBoxColMax - boundingBoxColMin)

        # Least Feret’s Diameter (LFD), Feret’s Width: The smallest distance between two parallel lines that do not intersect the image. TODO
        # Greatest Feret’s Diameter (GFD), Feret’s Length: The greatest distance between two parallel lines that do not intersect the image. TODO

        # Equivalent Circular Area Diameter (ECAD), Heywood’s Diameter: The diameter of a circle that has the same area as the image.
        listEquivalentCircularAreaDiameter.append(cv2.contourArea(contour))

        # Least Bounding Circle (LBC): The smallest circle that encloses the image. TODO
        (x0, y0), radius = cv2.minEnclosingCircle(contour)
        center = (int(x0), int(y0))
        #radius = int(radius)
        #img = cv2.circle(img, center, radius, (0,255,0), 2)
        listLBCRadius.append(radius)

        # Convex Perimeter (CPM): The perimeter of a convex curve circumscribing the image.
        convex_contour = cv2.convexHull(contour)
        listCPM.append(cv2.arcLength(convex_contour, True))

        # Equivalent Circular Perimeter Diameter (ECPD): The diameter of a circle that has the same perimeter of the image.
        imagePerimeter = 2 * (boundingBoxRowMax - boundingBoxRowMin) + 2 * (boundingBoxColMax - boundingBoxColMin)
        r = imagePerimeter / (2 * math.pi)
        listECPD.append(2*r)

        #Horizontal Martin’s Diameter (HMD): The length of a line at horizontal direction that divides the image into two equal halves. TODO
        #Vertical Martin’s Diameter (VMD): The length of a line at vertical direction that divides the image into two equal halves. TODO
        #Least Bounding Rectangle Width (LBRW): The width of the smallest rectangle that encloses the image. TODO
        #Least Bounding Rectangle Length (LBRL): The length of the smallest rectangle that encloses the image. TODO

        # Fiber Length (FL): The length of a rectangle that has the same area and perimeter as the image.
        # Fiber Width (FW): The width of a rectangle that has the same area and perimeter as the image.
        P = 2*(boundingBoxColMax - boundingBoxColMin) + 2*(boundingBoxRowMax - boundingBoxRowMin)
        A = props.area
        fiberLength = 0

        a = 1
        b = -P/2
        c = A
        d = b**2-4*a*c # discriminant

        if d < 0:
            print ("This equation has no real solution")
        elif d == 0:
            fiberLength = (-b+math.sqrt(b**2-4*a*c))/2*a
        else:
            fiberLength = (-b+math.sqrt((b**2)-(4*(a*c))))/(2*a) # TODO
            #x2 = (-b-math.sqrt((b**2)-(4*(a*c))))/(2*a)
        
        fiberWidth = P / 2 - fiberLength

        listFiberLength.append(fiberLength)
        listFiberWidth.append(fiberWidth)

        if (len(contour) > 4):
            # Fitted Ellipse
            ellipse = cv2.fitEllipse(contour)
            listFittedEllipse.append(ellipse)

            # Major Minor Axis Length
            (x, y), (Major, Minor), angle = cv2.fitEllipse(contour)
            listMajorAxisLength.append(Major)
            listMinorAxisLength.append(Minor)
        else:
            # Fitted Ellipse
            listFittedEllipse.append([])

            # Major Minor Axis Length
            listMajorAxisLength.append(0)
            listMinorAxisLength.append(0)

        # Kolla om de är konvexa (har “massa” i centerpunkten) TODO
        listIsConvex.append(cv2.isContourConvex(contour))
        
        listPerimeters.append(props.perimeter)
        listCentroids.append(props.local_centroid)

    if False:
        for idx, grain in enumerate(listGrainImages):

            # Add 1 pixel border around the grain
            grainNew = np.zeros((grain.shape[0]+2, grain.shape[1]+2))
            for x in range(grain.shape[0]):
                for y in range(grain.shape[1]):
                    grainNew[x+1][y+1] = grain[x][y]

            fig, ax = plt.subplots()
            ax.set_facecolor("black")
            
            #print("Local Maxi: " + str(local_maxi))
            grain = img_as_ubyte(grain)
            #(x0, y0, radius)= listLBCRadius[idx]
            #ax.plot(x0, y0, '.g', markersize=5)
            #ax.add_artist(plt.Circle((x0+1, y0+1), radius, color='g', fill=False, linestyle='dashed'))

            #grain = cv2.circle(grain, (int(x0), int(y0)), radius, (0,255,0), 2)
            ax.imshow(grain, cmap=plt.cm.gray)

            x0, y0 = listCentroids[idx]
            ax.plot(x0+1, y0+1, '.r', markersize=5)
            ax.add_artist(plt.Circle((x0+1, y0+1), listEquivalentCircularAreaDiameter[idx]/2.0, color='r', fill=False, linestyle='dashed'))

            plt.show()

    # Useful information / statistics
    print("Number of grains: " + str(len(regions)))

    if (DEBUG_PLOT):
        plotImages()

    plotData()

def preProcessImage(inputImage):
    ### Aquired image ###
    image = img_as_ubyte(inputImage)

    # Scale
    scale = 1
    image = cv2.resize(image,None,fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    listDebugImages.append(image)
    listDebugTitles.append('Scaled Image')

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #color.rgb2gray(image)
    listDebugImages.append(image)
    listDebugTitles.append('Grayscale Image')

    return image

def filterImage(inputImage):
    ### Filtering ###
    image = img_as_ubyte(inputImage)
    
    if HYBRID_FILTER:
        # Hybrid median filter
        cross = np.matrix([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
        xmask = np.matrix([[1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1]])
        center = np.matrix([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

        image = median(image, selem=cross)
        listDebugImages.append(image)
        listDebugTitles.append('Hybrid Median Filter 1')
        image = median(image, selem=xmask)
        listDebugImages.append(image)
        listDebugTitles.append('Hybrid Median Filter 2')
        image = median(image, selem=center)
        listDebugImages.append(image)
        listDebugTitles.append('Hybrid Median Filter 3')
    else:
        # Median filter
        image = median(image)
        listDebugImages.append(image)
        listDebugTitles.append('Median Filter')

    # Contrast filter
    image = enhance_contrast(image, disk(5))
    listDebugImages.append(image)
    listDebugTitles.append('Contrast Filter')
 
    return image

def segmentImage(inputImage):
    ### Segmenting ###
    #image = img_as_ubyte(inputImage)
    image = inputImage

    #image = cv2.medianBlur(image, 5)
    # Threshold
    #threshold = cv2.adaptiveThreshold(image, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 1)
    threshold = skimage.filters.threshold_otsu(image)
    #threshold = skimage.filters.threshold_local(image, 3)
    
    # Binary
    image = inputImage <= threshold
    listDebugImages.append(image)
    listDebugTitles.append('Binary Threshold')

    if (DEBUG_SEGMENTING):
        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1, adjustable='box-forced')
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')

        ax[0].imshow(inputImage, cmap=plt.cm.gray)
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].hist(inputImage.ravel(), bins=256)
        ax[1].set_title('Histogram')
        ax[1].axvline(threshold, color='r')

        ax[2].imshow(image, cmap=plt.cm.gray)
        ax[2].set_title('Thresholded')
        ax[2].axis('off')
        plt.show()

    return image

def editImage(inputImage):
    ### Editing ###
    global labels, distance
    image = inputImage

    # Fill interior TODO
    #seed = np.copy(image)
    #seed[-1:-1, -1:-1] = image.min()
    #mask = image
    #filled = reconstruction(seed, mask, method='erosion')
    #image = filled

    #seed = np.copy(image)
    #seed[1:-1, 1:-1] = image.min()
    #rec = reconstruction(seed, image, method='dilation')
    #image = rec

    # Remove objects TODO

    # Separate objects / Watershedding
    distance = ndi.distance_transform_edt(image)
    listDebugImages.append(distance)
    listDebugTitles.append('Distance')

    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
    listDebugImages.append(local_maxi)
    listDebugTitles.append('Local Maxi')

    markers = ndi.label(local_maxi)[0]

    labels = watershed(-distance, markers, mask=image)
    listDebugImages.append(labels)
    listDebugTitles.append('Watershed')

    if (DEBUG_EDITING):
        fig, axes = plt.subplots(ncols=4, figsize=(9, 3), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(imageOriginal, cmap=plt.cm.gray, interpolation='nearest')
        ax[0].set_title('Original image')
        ax[1].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax[1].set_title('Filled interior')
        ax[2].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
        ax[2].set_title('Distances')
        ax[3].imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
        ax[3].set_title('Separated objects')

        for a in ax:
            a.set_axis_off()

        fig.tight_layout()
        plt.show()

    image = labels
    return image

def loadImage():
    try:
        image = io.imread(FILENAME)
        listDebugImages.append(image)
        listDebugTitles.append('Original Image')
        return image
    except:
        print("No file named '" + FILENAME + "' found. Exiting program!")
        sys.exit()

def plotImages():

    subplotRows = 4
    subplotColumns = 3
    ax = plt.subplot(subplotRows, subplotColumns, 1)
    for idx, im in enumerate(listDebugImages):
        plt.subplot(subplotRows, subplotColumns, idx+1, sharex=ax, sharey=ax)
        plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        plt.title(listDebugTitles[idx])
        plt.axis('off')

    #fig, axes = plt.subplots(ncols=5, figsize=(9, 3), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    #ax = axes.ravel()
#
    #ax[0].imshow(imageOriginal)
    #ax[0].set_title('Original')
    #ax[1].imshow(imagePreProcessed, cmap=plt.cm.gray, interpolation='nearest')
    #ax[1].set_title('Aquired')
    #ax[2].imshow(imageFiltered, cmap=plt.cm.gray, interpolation='nearest')
    #ax[2].set_title('Filtered')
    #ax[3].imshow(imageSegmented, cmap=plt.cm.gray, interpolation='nearest')
    #ax[3].set_title('Segmented')
    #ax[4].imshow(imageEdited, cmap=plt.cm.gray, interpolation='nearest')
    #ax[4].set_title('Edited')
#
    ## Remove axis
    #for a in ax:
    #   a.set_axis_off()
#
    #fig.tight_layout()
    plt.show()

def plotData():
    plt.subplot(131), plt.hist(listAreas, bins=100, density=True, histtype='step', cumulative=True)
    plt.subplot(131), plt.hist(listAreas, bins=100, density=True, histtype='step', cumulative=-1)
    plt.subplot(131), plt.title('Histogram Area')
    plt.subplot(131), plt.xlabel('Area Size')
    plt.subplot(131), plt.ylabel('Occurence')
    plt.subplot(131), plt.grid(True)
    plt.subplot(132), plt.hist(listEquivalentCircularAreaDiameter, bins=100, density=True, histtype='step', cumulative=True)
    plt.subplot(132), plt.hist(listEquivalentCircularAreaDiameter, bins=100, density=True, histtype='step', cumulative=-1)
    plt.subplot(132), plt.title('Histogram Equivalent Circular Area Diameter')
    plt.subplot(132), plt.xlabel('Equivalent Circular Area Diameter')
    plt.subplot(132), plt.ylabel('Occurence')
    plt.subplot(132), plt.grid(True)
    plt.subplot(133), plt.hist(listLBCRadius, bins=100, density=True, histtype='step', cumulative=True)
    plt.subplot(133), plt.hist(listLBCRadius, bins=100, density=True, histtype='step', cumulative=-1)
    plt.subplot(133), plt.title('Histogram Equivalent Circular Area Diameter')
    plt.subplot(133), plt.xlabel('Equivalent Circular Area Diameter')
    plt.subplot(133), plt.ylabel('Occurence')
    plt.subplot(133), plt.grid(True)

    plt.show()








##### MAIN ####
def main():
    global DEBUG_FILTERING, DEBUG_SEGMENTING, DEBUG_EDITING, DEBUG_PLOT
    DEBUG_FILTERING = False
    DEBUG_SEGMENTING = False
    DEBUG_EDITING = False
    DEBUG_PLOT = True

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:d", ["help", "debug"])
    except getopt.GetoptError as err:
        print(err)
        print("error")
        sys.exit(2)

    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print("--help")
            sys.exit(0)
        if o in ("-d", "--debug"):
            print("--debug")
            DEBUG_FILTERING = True
            DEBUG_SEGMENTING = True
            DEBUG_EDITING = True
            DEBUG_PLOT = True

    # process arguments
    for arg in args:
        process(arg) # process() is defined elsewhere

    analyze()

def process(arg):
    print(arg)

if __name__ == "__main__":
    main()