import sys
import getopt
import math
import numpy as np
import cv2
import statistics
import lmfit
import scipy
from scipy.interpolate import interp1d

from xlsxwriter import Workbook #, easyxf

from PIL import Image
from scipy import ndimage as ndi
import matplotlib # matplotlib.rcParams.update({'font.size': 30})
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
from skimage.morphology import remove_small_holes
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate

### Constants
global FILENAME
FILENAME = 'images/image/latest0.jpg'
global X50REF
global X503TEMP
global CDF_PERCENTILE_REF
X50REF = 1
X503TEMP = 1
CDF_PERCENTILE_REF = [2.55,  3.04,  3.50,  3.94,  4.54,  5.27,  5.91,  6.60,
                   7.30,  7.88,  8.46,  9.20,  10.24, 11.33, 12.45, 14.01,
                   16.09, 18.92, 22.46, 28.84, 39.86, 55.84, 74.26, 89.83,
                   98.02, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
CDF_X0_REF = [18, 22, 26, 30, 36, 44, 52, 62, # TODO Histogram och standard variation
            74, 86, 100, 120, 150, 180, 210, 250,
            300, 360, 420, 500, 600, 720, 860, 1020,
            1220, 1460, 1740, 2060, 2460, 2940, 3500]
# TODO um / pixel microMeterPerPixel = 20.0 # 19.8
MICRO_METER_PER_PIXEL = 470/22 #17.4 #18.15 * 2.0#19.8 #17.39
DIAMETER_THRESHOLD_LOW = 2 * MICRO_METER_PER_PIXEL
DIAMETER_THRESHOLD_HIGH = 5000#1460

# Excel setup
wb = Workbook('output.xlsx')
excel_sheet = wb.add_worksheet() #wb.add_sheet('Test all')
formatRed = wb.add_format({
    'bold': 'true', 
    'bg_color': 'red'
})
formatYellow = wb.add_format({
    'bold': 'true', 
    'bg_color': 'yellow'
})
formatGreen = wb.add_format({
    'bold': 'true', 
    'bg_color': 'green'
})
global EXCEL_ROW
EXCEL_ROW = 1

excel_sheet.write(0, 0, 'Micrometer / Pixel')
excel_sheet.write(1, 0, '=1.0/' + str(MICRO_METER_PER_PIXEL))
excel_sheet.write(0, 1, 'Filename')
excel_sheet.write(0, 2, 'Reference x50')
excel_sheet.write(0, 3, '3TEMP x50')
excel_sheet.write(0, 4, '3TEMP Deviation')
#excel_sheet.write(0, 5, 'Heights x50')
#excel_sheet.write(0, 6, 'Deviation')
#excel_sheet.write(0, 7, 'Standard Deviation')
#excel_sheet.write(0, 8, 'PValue')
#excel_sheet.write(0, 9, 'Widths x50')
#excel_sheet.write(0, 10, 'Deviation')
#excel_sheet.write(0, 11, 'Standard Deviation')
#excel_sheet.write(0, 12, 'PValue')
#excel_sheet.write(0, 13, 'Least Feret Diameter x50')
#excel_sheet.write(0, 14, 'Deviation')
#excel_sheet.write(0, 15, 'Standard Deviation')
#excel_sheet.write(0, 16, 'PValue')
#excel_sheet.write(0, 17, 'Greatest Feret Diameter x50')
#excel_sheet.write(0, 18, 'Deviation')
#excel_sheet.write(0, 19, 'Standard Deviation')
#excel_sheet.write(0, 20, 'PValue')
#excel_sheet.write(0, 21, 'Mean Feret Diameter x50')
#excel_sheet.write(0, 22, 'Deviation')
#excel_sheet.write(0, 23, 'Standard Deviation')
#excel_sheet.write(0, 24, 'PValue')
#excel_sheet.write(0, 25, 'Equivalent Circle Perimeter Diameter x50')
#excel_sheet.write(0, 26, 'Deviation')
#excel_sheet.write(0, 27, 'Standard Deviation')
#excel_sheet.write(0, 28, 'PValue')
#excel_sheet.write(0, 29, 'Equivalent Circle Area Diameter x50')
#excel_sheet.write(0, 30, 'Deviation')
#excel_sheet.write(0, 31, 'Standard Deviation')
#excel_sheet.write(0, 32, 'PValue')
#excel_sheet.write(0, 33, 'Least Bounding Circle Diameter x50')
#excel_sheet.write(0, 34, 'Deviation')
#excel_sheet.write(0, 35, 'Standard Deviation')
#excel_sheet.write(0, 36, 'PValue')
#excel_sheet.write(0, 37, 'Horizontal Martin Diameter x50')
#excel_sheet.write(0, 38, 'Deviation')
#excel_sheet.write(0, 39, 'Standard Deviation')
#excel_sheet.write(0, 40, 'PValue')
#excel_sheet.write(0, 41, 'Vertical Martin Diameter x50')
#excel_sheet.write(0, 42, 'Deviation')
#excel_sheet.write(0, 43, 'Standard Deviation')
#excel_sheet.write(0, 44, 'PValue')
#excel_sheet.write(0, 45, 'Least Bounding Rectagle Width x50')
#excel_sheet.write(0, 46, 'Deviation')
#excel_sheet.write(0, 47, 'Standard Deviation')
#excel_sheet.write(0, 48, 'PValue')
#excel_sheet.write(0, 49, 'Least Bounding Rectangle Length x50')
#excel_sheet.write(0, 50, 'Deviation')
#excel_sheet.write(0, 51, 'Standard Deviation')
#excel_sheet.write(0, 52, 'PValue')
#excel_sheet.write(0, 53, 'Fiber Length x50')
#excel_sheet.write(0, 54, 'Deviation')
#excel_sheet.write(0, 55, 'Standard Deviation')
#excel_sheet.write(0, 56, 'PValue')
#excel_sheet.write(0, 57, 'Fiber Width x50')
#excel_sheet.write(0, 58, 'Deviation')
#excel_sheet.write(0, 59, 'Standard Deviation')
#excel_sheet.write(0, 60, 'PValue')
#excel_sheet.write(0, 61, 'Major Axis Length x50')
#excel_sheet.write(0, 62, 'Deviation')
#excel_sheet.write(0, 63, 'Standard Deviation')
#excel_sheet.write(0, 64, 'PValue')
#excel_sheet.write(0, 65, 'Minor Axis Length x50')
#excel_sheet.write(0, 66, 'Deviation')
#excel_sheet.write(0, 67, 'Standard Deviation')
#excel_sheet.write(0, 68, 'PValue')

global DEBUG_FILTERING
global DEBUG_SEGMENTING
global DEBUG_EDITING
global DEBUG_PLOT
global DEBUG_DATA
global DEBUG_STUDENTS_TTEST
global DEBUG_SAVE

global DEBUG_HEIGHTS
global DEBUG_WIDHTS
global DEBUG_LFD
global DEBUG_GFD
global DEBUG_MFD
global DEBUG_ECPD
global DEBUG_ECAD
global DEBUG_LBCD
global DEBUG_HMD
global DEBUG_VMD
global DEBUG_LBRW
global DEBUG_LBRL
global DEBUG_FL
global DEBUG_FW
global DEBUG_MAJOR_AXIS
global DEBUG_MINOR_AXIS

HYBRID_FILTER = False
DEBUG_WATERSHED = True

global imageOriginal
global imagePreProcessed
global imageFiltered
global imageSegmented
global imageEdited
global labels
global distance

global listDebugImages, listDebugImageTitles, listDebugMeasurementTitles

global listGrainImages
global listHeights
global listWidths
global listLFDiameter
global listGFDiameter
global listMFDiameter
global listBoundingBoxAreas
global listAreas
global listCPM
global listECPDiameter
global listPerimeters
global listCentroids
global listECADiameter
global listLBCDiameter
global listHMDiameter
global listVMDiameter
global listLBRW
global listLBRL
global listFiberLength
global listFiberWidth
global listFittedEllipse
global listMajorAxisLength
global listMinorAxisLength
global listIsConvex

def analyze():
    global imageOriginal, imagePreProcessed, imageFiltered, imageSegmented, imageEdited
    global listDebugImages, listDebugImageTitles, listDebugMeasurementTitles
    global listGrainImages, listHeights, listWidths, listLFDiameter, listGFDiameter, listMFDiameter, listBoundingBoxAreas, listAreas, listCPM, listECPDiameter, listPerimeters, listCentroids, listECADiameter, listLBCDiameter, listHMDiameter, listVMDiameter, listLBRW, listLBRL, listFiberLength, listFiberWidth, listFittedEllipse, listMajorAxisLength, listMinorAxisLength, listIsConvex
 
    listDebugImages = list()
    listDebugImageTitles = list()
    listDebugMeasurementTitles = list()

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

    # Edge Detection
    #image_tmp = img_as_ubyte(imageEdited)
    #imageEdges = auto_canny(image_tmp)
    #plt.plot(imageEdited)
    #plt.show()

    ### Measure ###
    # TODO
    label_img = ndi.label(imageEdited)[0] #label(imageEdited)
    #local_maxi = peak_local_max(imageEdited, indices=False, footprint=np.ones((3, 3)), labels=label_img, num_peaks_per_label=1)
    #print("Local Maxi: " + str(local_maxi) + ")")
    regions = regionprops(label_img)

    listGrainImages = list()
    listHeights = list()
    listWidths = list()
    listBoundingBoxAreas = list()
    listAreas = list()
    listLFDiameter = list()
    listGFDiameter = list()
    listMFDiameter = list()
    listCPM = list()
    listECPDiameter = list()
    listPerimeters = list()
    listCentroids = list()
    listECADiameter = list()
    listLBCDiameter = list()
    listHMDiameter = list()
    listVMDiameter = list()
    listLBRW = list()
    listLBRL = list()
    listFiberLength = list()
    listFiberWidth = list()
    listFittedEllipse = list()
    listMajorAxisLength = list()
    listMinorAxisLength = list()
    listIsConvex = list()
    count = 0

    fig, ax = plt.subplots()

    for props in regions:

        if False:
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

        
        if len(contour) > 0 and cv2.arcLength(contour, True) > 0.0 and cv2.contourArea(contour) > 0.0: #and props.area > 25 and props.area < 2500:
            listGrainImages.append(props.image)

            # Height of boundingbox
            if (DEBUG_HEIGHTS):
                height = (boundingBoxColMax - boundingBoxColMin) * MICRO_METER_PER_PIXEL
                listHeights.append(height)

            # Width of boundingbox
            if (DEBUG_WIDHTS):
                width = (boundingBoxRowMax - boundingBoxRowMin) * MICRO_METER_PER_PIXEL
                listWidths.append(width)

            # Area
            listAreas.append(props.area * MICRO_METER_PER_PIXEL * MICRO_METER_PER_PIXEL)
            #listBoundingBoxAreas.append(width * height)

            # Least Feret’s Diameter (LFD), Feret’s Width: The smallest distance between two parallel lines that do not intersect the image.
            # Greatest Feret’s Diameter (GFD), Feret’s Length: The greatest distance between two parallel lines that do not intersect the image.
            if (DEBUG_LFD or DEBUG_GFD or DEBUG_MFD):
                (LFDiameter, GFDiameter) = feretsDiameters(contour)
                listLFDiameter.append(LFDiameter * MICRO_METER_PER_PIXEL)
                listGFDiameter.append(GFDiameter * MICRO_METER_PER_PIXEL)
                listMFDiameter.append((GFDiameter + LFDiameter) / 2 * MICRO_METER_PER_PIXEL)

            # Equivalent Circular Area Diameter (ECAD), Heywood’s Diameter: The diameter of a circle that has the same area as the image.
            if (DEBUG_ECAD):
                diameter = math.sqrt(cv2.contourArea(contour) / math.pi) * 2 * MICRO_METER_PER_PIXEL
                listECADiameter.append(diameter)

            # Least Bounding Circle (LBC): The smallest circle that encloses the image.
            if (DEBUG_LBCD):
                (x0, y0), radius = cv2.minEnclosingCircle(contour)
                center = (int(x0), int(y0))
                #radius = int(radius)
                #img = cv2.circle(img, center, radius, (0,255,0), 2)
                listLBCDiameter.append(radius * 2 * MICRO_METER_PER_PIXEL)

            # Convex Perimeter (CPM): The perimeter of a convex curve circumscribing the image.
            convex_contour = cv2.convexHull(contour)
            listCPM.append(cv2.arcLength(convex_contour, True) * MICRO_METER_PER_PIXEL)

            # Equivalent Circular Perimeter Diameter (ECPD): The diameter of a circle that has the same perimeter of the image.
            if (DEBUG_ECPD):
                imagePerimeter = cv2.arcLength(contour, True)
                #imagePerimeter = 2 * (boundingBoxRowMax - boundingBoxRowMin) + 2 * (boundingBoxColMax - boundingBoxColMin) * MICRO_METER_PER_PIXEL
                radius = imagePerimeter / (2 * math.pi)
                listECPDiameter.append(radius * 2 * MICRO_METER_PER_PIXEL)

            # Horizontal Martin’s Diameter (HMD): The length of a line at horizontal direction that divides the image into two equal halves.
            # Vertical Martin’s Diameter (VMD): The length of a line at vertical direction that divides the image into two equal halves.
            if (DEBUG_HMD or DEBUG_VMD):
                cutArea = props.area / 2
                accArea = 0
                horizontalBreak = 0
                verticalBreak = 0
                for x in range(len(props.image)):
                    for y in range(len(props.image[0])):
                        if props.image[x][y] > 0:
                            accArea += 1
                        if accArea >= cutArea:
                            horizontalBreak = x
                            break

                accArea = 0
                for y in range(len(props.image[0])):
                    for x in range(len(props.image)):
                        if props.image[x][y] > 0:
                            accArea += 1
                        if accArea >= cutArea:
                            verticalBreak = y
                            break

                horizontalDiameter = 0
                verticalDiameter = 0
                for pixel in props.image[x]:
                    if pixel > 0:
                        horizontalDiameter += 1

                for x, _ in enumerate(props.image):
                    if props.image[x][y] > 0:
                        verticalDiameter += 1

                listHMDiameter.append(horizontalDiameter * MICRO_METER_PER_PIXEL)
                listVMDiameter.append(verticalDiameter * MICRO_METER_PER_PIXEL)

            # Least Bounding Rectangle Width (LBRW): The width of the smallest rectangle that encloses the image.
            # Least Bounding Rectangle Length (LBRL): The length of the smallest rectangle that encloses the image.
            if (DEBUG_LBRW or DEBUG_LBRL):
                (x, y), (w, h), r = cv2.minAreaRect(contour)
                listLBRW.append(w * MICRO_METER_PER_PIXEL)
                listLBRL.append(h * MICRO_METER_PER_PIXEL)

            # Fiber Length (FL): The length of a rectangle that has the same area and perimeter as the image.
            # Fiber Width (FW): The width of a rectangle that has the same area and perimeter as the image.
            if (DEBUG_FL or DEBUG_FW):
                P = 2 * (boundingBoxColMax - boundingBoxColMin) + 2 * (boundingBoxRowMax - boundingBoxRowMin)
                #print(str(x1) + " " + str(x2) + " " + str(y1) + " " + str(y2))
                #P = 2 * w + 2 * h
                A = props.area
                fiberLength = 0

                a = 1
                b = -P/2
                c = A
                d = b**2-4*a*c # discriminant

                if d < 0:
                    print("This equation has no real solution for " + str(P))
                    #print("a: " + str(a))
                    #print("b: " + str(b))
                    #print("c: " + str(c))
                    #print("d: " + str(d))
                elif d == 0:
                    fiberLength = (-b + math.sqrt(b**2 - 4*a*c)) / (2 * a)
                else:
                    fiberLength = (-b + math.sqrt(b**2 - 4*a*c)) / (2 * a) # TODO
                    #x2 = (-b-math.sqrt((b**2)-(4*(a*c))))/(2*a)
                
                fiberWidth = P / 2 - fiberLength

                listFiberLength.append(fiberLength * MICRO_METER_PER_PIXEL)
                listFiberWidth.append(fiberWidth * MICRO_METER_PER_PIXEL)

            if (DEBUG_MAJOR_AXIS or DEBUG_MINOR_AXIS):
                if (len(contour) > 4):
                    # Fitted Ellipse
                    ellipse = cv2.fitEllipse(contour)
                    listFittedEllipse.append(ellipse)

                    # Major Minor Axis Length
                    (x, y), (Major, Minor), angle = cv2.fitEllipse(contour)
                    listMajorAxisLength.append(Major * MICRO_METER_PER_PIXEL)
                    listMinorAxisLength.append(Minor * MICRO_METER_PER_PIXEL)
                else:
                    # Fitted Ellipse
                    listFittedEllipse.append([])

                    # Major Minor Axis Length
                    listMajorAxisLength.append(0)
                    listMinorAxisLength.append(0)

            # Kolla om de är konvexa (har “massa” i centerpunkten) TODO
            listIsConvex.append(cv2.isContourConvex(contour))
            
            listPerimeters.append(props.perimeter * MICRO_METER_PER_PIXEL)
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
            #(x0, y0, radius)= listLBCDiameter[idx]
            #ax.plot(x0, y0, '.g', markersize=5)
            #ax.add_artist(plt.Circle((x0+1, y0+1), radius, color='g', fill=False, linestyle='dashed'))

            #grain = cv2.circle(grain, (int(x0), int(y0)), radius, (0,255,0), 2)
            ax.imshow(grain, cmap=plt.cm.gray)

            x0, y0 = listCentroids[idx]
            #ax.plot(x0+1, y0+1, '.r', markersize=5)
            #ax.add_artist(plt.Circle((x0+1, y0+1), listECADiameter[idx]/2.0, color='r', fill=False, linestyle='dashed'))

            plt.show()


    listMeasures = list()
    if (DEBUG_HEIGHTS):
        listHeights.sort()
        listMeasures.append(np.array(listHeights))
        listDebugMeasurementTitles.append("Height")
    if (DEBUG_WIDHTS):
        listWidths.sort()
        listMeasures.append(np.array(listWidths))
        listDebugMeasurementTitles.append("Width")
    if (DEBUG_LFD):
        listLFDiameter.sort()
        listMeasures.append(np.array(listLFDiameter))
        listDebugMeasurementTitles.append("Least Feret Diameter")
    if (DEBUG_GFD):
        listGFDiameter.sort()
        listMeasures.append(np.array(listGFDiameter))
        listDebugMeasurementTitles.append("Greatest Feret Diameter")
    if (DEBUG_MFD):
        listMFDiameter.sort()
        listMeasures.append(np.array(listMFDiameter))
        listDebugMeasurementTitles.append("Mean Feret Diameter")
    if (DEBUG_ECPD):
        listECPDiameter.sort()
        listMeasures.append(np.array(listECPDiameter))
        listDebugMeasurementTitles.append("Equivalent Circle Perimeter Diameter")
    if (DEBUG_ECAD):
        listECADiameter.sort()
        listMeasures.append(np.array(listECADiameter))
        listDebugMeasurementTitles.append("Equivalent Circle Area Diameter")
    if (DEBUG_LBCD):
        listLBCDiameter.sort()
        listMeasures.append(np.array(listLBCDiameter))
        listDebugMeasurementTitles.append("Least Bounding Circle Diameter")
    if (DEBUG_HMD):
        listHMDiameter.sort()
        listMeasures.append(np.array(listHMDiameter))
        listDebugMeasurementTitles.append("Horizontal Martin Diameter")
    if (DEBUG_VMD):
        listVMDiameter.sort()
        listMeasures.append(np.array(listVMDiameter))
        listDebugMeasurementTitles.append("Vertical Martin Diameter")
    if (DEBUG_LBRW):
        listLBRW.sort()
        listMeasures.append(np.array(listLBRW))
        listDebugMeasurementTitles.append("Least Bounding Rectangle Width")
    if (DEBUG_LBRL):
        listLBRL.sort()
        listMeasures.append(np.array(listLBRL))
        listDebugMeasurementTitles.append("Least Bounding Rectangle Height")
    if (DEBUG_FL):
        listFiberLength.sort()
        listMeasures.append(np.array(listFiberLength))
        listDebugMeasurementTitles.append("Fiber Length")
    if (DEBUG_FW):
        listFiberWidth.sort()
        listMeasures.append(np.array(listFiberWidth))
        listDebugMeasurementTitles.append("Fiber Width")
    if (DEBUG_MAJOR_AXIS):
        listMajorAxisLength.sort()
        listMeasures.append(np.array(listMajorAxisLength))
        listDebugMeasurementTitles.append("Major Axis")
    if (DEBUG_MINOR_AXIS):
        listMinorAxisLength.sort()
        listMeasures.append(np.array(listMinorAxisLength))
        listDebugMeasurementTitles.append("Minor Axis")

    if (DEBUG_PLOT):
        plotImages()

    if (DEBUG_DATA):
        plotData()

    # Weight data
    #listMeasures = applyWeight(listMeasures)

    temp3Deviation = X503TEMP/float(X50REF)*100 - 100


    NUMBER_OF_DATA_ENTRIES = 4
    NUMBER_OF_FIXED_ENTRIES = 5
    excel_sheet.write(EXCEL_ROW, 1, str(FILENAME))
    excel_sheet.write(EXCEL_ROW, 2, str(X50REF))
    excel_sheet.write(EXCEL_ROW, 3, str(X503TEMP))
    if math.fabs(temp3Deviation) < 10:
        excel_sheet.write(EXCEL_ROW, 4, temp3Deviation, formatGreen)
    else:
        excel_sheet.write(EXCEL_ROW, 4, temp3Deviation, formatRed)

    for idx, l in enumerate(listDebugMeasurementTitles):
        excel_sheet.write(0, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES, str(l) + ' x50')
        excel_sheet.write(0, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+1, 'Deviation')
        excel_sheet.write(0, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+2, 'Standard Deviation')
        excel_sheet.write(0, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+3, 'PValue')

    for idx, measure in enumerate(listMeasures):
        #measureDeviation = measure/float(X50REF)*100 - 100
        x50measure = np.percentile(measure, 50)
        (_, t_test_pval, _, wilcoxon_pval) = students_ttest_wilcoxon(measure, listDebugMeasurementTitles[idx])

        print("measure: " + str(measure))
        print("x50measure: " + str(x50measure))
        excel_sheet.write(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES, float(x50measure)) # TODO
        excel_sheet.write(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+1, str('=ABS($A$2*OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())),0,-1)/OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, ' + str(-4-idx*NUMBER_OF_DATA_ENTRIES) + ')*100-100)'))
        excel_sheet.write(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+2, float(statistics.pstdev(measure)))
        excel_sheet.write(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+3, float(t_test_pval))

        excel_sheet.conditional_format(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+1, EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+1, {'type': 'cell',
                                                   'criteria': 'between',
                                                   'minimum': -10,
                                                   'maximum': 10,
                                                   'format': formatGreen})
        excel_sheet.conditional_format(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+1, EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+1, {'type': 'cell',
                                                   'criteria': 'between',
                                                   'minimum': '-ABS(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, ' + str(-2-idx*NUMBER_OF_DATA_ENTRIES) + '))',
                                                   'maximum': 'ABS(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, ' + str(-2-idx*NUMBER_OF_DATA_ENTRIES) + '))',
                                                   'format': formatYellow})
        excel_sheet.conditional_format(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+1, EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+1, {'type': 'cell',
                                                   'criteria': 'not between',
                                                   'minimum': '-ABS(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, ' + str(-2-idx*NUMBER_OF_DATA_ENTRIES) + '))',
                                                   'maximum': 'ABS(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, ' + str(-2-idx*NUMBER_OF_DATA_ENTRIES) + '))',
                                                   'format': formatRed})

    #Q1 = (arrayListMeasureLength[-1] + 1) / 4
    #Q2 = 2 * (arrayListMeasureLength[-1] + 1) / 4
    #Q3 = 3 * (arrayListMeasureLength[-1] + 1) / 4

    #f = np.fft.fft2(imagePreProcessed)
    #fshift = np.fft.fftshift(f)
    #magnitude_spectrum = 20*np.log(np.abs(fshift))
    #plt.plot(magnitude_spectrum)
    #plt.show()

    #laplacian = cv2.Laplacian(imagePreProcessed, 32)
    #bluriness = magnitude_spectrum.var()
    #print("cv2.CV_64F = " + str(cv2.CV_64F))

    print("\n\n\n##### " + str(FILENAME) + " # " + str(X50REF) + " #####")
    #print("3TEMP x50: " + str(X503TEMP) + "   " + "Deviation %: " + str(X503TEMP/float(X50REF)*100 - 100))
    #print("      x50: " + str(x50) + "   " + "Deviation %: " + str(x50/float(X50REF)*100 - 100))

    # Useful information / statistics
    print("Total number of grains: " + str(len(regions)))
    #print("Number of dirt grains:" + str(count))
    #print("Number of dirt grains %:" + str(100*count/len(regions)))

    #Sauter3 = 0
    #Sauter2 = 0
    #for a in listECADiameter:
    #    Sauter3 += a**3
    #    Sauter2 += a**2
    #Sauter32 = Sauter3 / Sauter2

    #print("Sauter32: " + str(Sauter32))

    #print("")
    #print(listGFDiameter)
    #print(listECADiameter)
    #print("Blur Score: " + str(bluriness))
    #print("Measured x10: " + str(x10))
    #print("Measured x16: " + str(x16))
    #print("Measured x50: " + str(x50))
    #print("Measured x84: " + str(x84))
    #print("Measured x90: " + str(x90))
    #print("Measured x99: " + str(x99))
    
    #plt.plot(laplacian)
    #plt.show()

    #print("Standard Deviation: " + str(statistics.pstdev(listFiberLength)))
    #print("Variance: " + str(statistics.pvariance(listFiberLength)))
    #print("Q1: " + str(Q1))
    #print("Q2: " + str(Q2))
    #print("Q3: " + str(Q3))

def preProcessImage(inputImage):
    ### Aquired image ###
    image = img_as_ubyte(inputImage)

    # Scale
    scale = 1
    image = cv2.resize(image,None,fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    listDebugImages.append(image)
    listDebugImageTitles.append('Scaled Image')

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #color.rgb2gray(image)
    listDebugImages.append(image)
    listDebugImageTitles.append('Grayscale Image')

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
        listDebugImageTitles.append('Hybrid Median Filter 1')
        image = median(image, selem=xmask)
        listDebugImages.append(image)
        listDebugImageTitles.append('Hybrid Median Filter 2')
        image = median(image, selem=center)
        listDebugImages.append(image)
        listDebugImageTitles.append('Hybrid Median Filter 3')
    else:
        # Median filter
        image = median(image)
        #image = median(image)
        #image = median(image)
        listDebugImages.append(image)
        listDebugImageTitles.append('Median Filter')

    # Contrast filter
    image = enhance_contrast(image, disk(5))
    listDebugImages.append(image)
    listDebugImageTitles.append('Contrast Filter')
 
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
    image = inputImage <= threshold#threshold #TODO Får inte samma värde som Anders, belysningen räknas in här
    listDebugImages.append(image)
    listDebugImageTitles.append('Binary Threshold')

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
        #plt.hist(inputImage.ravel(), bins=256)
        #plt.title('Histogram')
        #plt.axvline(threshold, color='r')
        plt.show()

    return image

def editImage(inputImage):
    ### Editing ###
    global labels, distance
    image = img_as_ubyte(inputImage)

    #kernel = np.ones((3,3),np.uint8)
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    #listDebugImages.append(image)
    #listDebugImageTitles.append('Opening')

    # Fill interior TODO
    #seed = np.copy(image)
    #seed[-1:-1, -1:-1] = image.max()
    #mask = image
    #filled = reconstruction(seed, mask, method='erosion')
    #image = filled
    #listDebugImages.append(image)
    #listDebugImageTitles.append('Erosion')

    #seed = np.copy(image)
    #seed[1:-1, 1:-1] = image.min()
    #rec = reconstruction(seed, image, method='dilation')
    #image = rec
    #listDebugImages.append(image)
    #listDebugImageTitles.append('Dilation')

    # Remove small holes TODO
    image = remove_small_holes(image, 7)
    listDebugImages.append(image)
    listDebugImageTitles.append('Remove small holes')

    # Separate objects / Watershedding
    distance = ndi.distance_transform_edt(image)
    listDebugImages.append(distance)
    listDebugImageTitles.append('Distance')

    #labels_tmp = label(image, connectivity=1)
    local_maxi = peak_local_max(distance, min_distance=10, indices=False, labels=image) #peak_local_max(distance, min_distance=100, indices=False, footprint=np.ones((3, 3)), labels=image)
    listDebugImages.append(local_maxi)
    listDebugImageTitles.append('Local Maxi')

    markers = ndi.label(local_maxi)[0]


    #labels = watershed(-distance, markers)
    #segmentation = ndi.binary_fill_holes(labels - 5)
    labels_new, _ = ndi.label(image)

    labels = watershed(-distance, markers, mask=labels_new, compactness=1.0, watershed_line=True)
    listDebugImages.append(labels)
    listDebugImageTitles.append('Watershed')

    if (DEBUG_EDITING):
        fig, axes = plt.subplots(ncols=4, figsize=(9, 3), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(imageOriginal, cmap=plt.cm.gray, interpolation='nearest')
        ax[0].set_title('Original image')
        ax[1].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax[1].set_title('Filled interior')
        ax[2].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
        ax[2].set_title('Distances')
        for idx, col in enumerate(labels):
            for idy, row in enumerate(col):
                if labels[idx][idy] > 0.0:
                    labels[idx][idy] = 0.0
                else:
                    labels[idx][idy] = 255
        ax[3].imshow(labels, cmap=plt.cm.binary, interpolation='nearest')
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
        listDebugImageTitles.append('Original Image')
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
        plt.title(listDebugImageTitles[idx])
        plt.axis('off')
        if DEBUG_SAVE:
            #io.imsave(str("output_images/" + listDebugImageTitles[idx]), im)
            #scipy.misc.imsave(str("output_images/" + listDebugImageTitles[idx] + ".jpg"), im)
            s = Image.fromarray(im).convert('RGB')
            s.save(str("output_images/" + listDebugImageTitles[idx] + ".jpg"))


    for im in listDebugImages:
        plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()

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

    #plt.imshow(image)
    #plt.show()

def plotData():

    # Compact Data Plot
    listHeights.sort()
    listWidths.sort()
    listLFDiameter.sort()
    listGFDiameter.sort()
    listMFDiameter.sort()
    listECPDiameter.sort()
    listLBCDiameter.sort()
    listHMDiameter.sort()
    listVMDiameter.sort()
    listLBRW.sort()
    listLBRL.sort()
    listFiberLength.sort()
    listFiberWidth.sort()
    listMajorAxisLength.sort()
    listMinorAxisLength.sort()

    showAsCumulative = True
    if (DEBUG_HEIGHTS):
        plt.hist(listHeights, bins=1000000, weights=[sphereVolume(d) for d in listHeights], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_WIDHTS):
        plt.hist(listWidths, bins=1000000, weights=[sphereVolume(d) for d in listWidths], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_LFD):
        plt.hist(listLFDiameter, bins=1000000, weights=[sphereVolume(d) for d in listLFDiameter], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_GFD):
        plt.hist(listGFDiameter, bins=1000000, weights=[sphereVolume(d) for d in listGFDiameter], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_MFD):
        plt.hist(listMFDiameter, bins=1000000, weights=[sphereVolume(d) for d in listMFDiameter], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_ECPD):
        plt.hist(listECPDiameter, bins=1000000, weights=[sphereVolume(d) for d in listECPDiameter], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_LBCD):
        plt.hist(listLBCDiameter, bins=1000000, weights=[sphereVolume(d) for d in listLBCDiameter], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_HMD):
        plt.hist(listHMDiameter, bins=1000000, weights=[sphereVolume(d) for d in listHMDiameter], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_VMD):
        plt.hist(listVMDiameter, bins=1000000, weights=[sphereVolume(d) for d in listVMDiameter], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_LBRW):
        plt.hist(listLBRW, bins=1000000, weights=[sphereVolume(d) for d in listLBRW], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_LBRL):
        plt.hist(listLBRL, bins=1000000, weights=[sphereVolume(d) for d in listLBRL], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_FL):
        plt.hist(listFiberLength, bins=1000000, weights=[sphereVolume(d) for d in listFiberLength], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_FW):
        plt.hist(listFiberWidth, bins=1000000, weights=[sphereVolume(d) for d in listFiberWidth], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_MAJOR_AXIS):
        plt.hist(listMajorAxisLength, bins=1000000, weights=[sphereVolume(d) for d in listMajorAxisLength], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    if (DEBUG_MINOR_AXIS):
        plt.hist(listMinorAxisLength, bins=1000000, weights=[sphereVolume(d) for d in listMinorAxisLength], density=True, histtype='step', linestyle='solid', cumulative=showAsCumulative)
    plt.title('Diameters')
    plt.legend(listDebugMeasurementTitles, loc='upper left')
    plt.axvline(x=float(X50REF), color='red')
    plt.axhline(y=0.5, color='red')
    plt.grid(True)
    plt.xscale('log')
    plt.show()

    #pars = lmfit.Parameters()
    #pars.add_many(('a', 0.1), ('b', 1))
    #mini = lmfit.Minimizer(listECPDiameter, pars)
    #result = mini.minimize()
    #print(lmfit.fit_report(result.params))
    
    #for idx, x in enumerate(listECPDiameter):
    #    listECPDiameter[idx] = (x / 2.0)**2 * math.pi
    #mid, low, high = mean_confidence_interval(listECPDiameter, confidence=0.70)
    #print(low, mid, high)

    #listECADiameter.sort()
    #for idx, x in enumerate(listECADiameter):
    #    listECADiameter[idx] = (x / 2.0)**2 * math.pi
    #listLBCDiameter.sort()
    #for idx, x in enumerate(listLBCDiameter):
    #    listLBCDiameter[idx] = (x / 2.0)**2 * math.pi

    #listAreas
    #stdev = statistics.pstdev(listAreas)
    #variance = statistics.pvariance(listAreas)
    #plt.subplot(211), plt.hist(listAreas, bins=10000, density=True, histtype='step', cumulative=True)
    #plt.subplot(211), plt.hist(listAreas, bins=10000, density=True, histtype='step', cumulative=-1)
    #plt.subplot(211), plt.title('Histogram Area\n' + 'Standard Deviation: ' + str(stdev) + '\n' + 'Variance: ' + str(variance))
    #plt.subplot(211), plt.xlabel('μm^2')
    #plt.subplot(211), plt.ylabel('Occurence')
    #plt.subplot(211), plt.grid(True)

    #listBoundingBoxAreas
    #plt.subplot(212), plt.hist(listBoundingBoxAreas, bins=10000, density=True, histtype='step', cumulative=True)
    #plt.subplot(212), plt.hist(listBoundingBoxAreas, bins=10000, density=True, histtype='step', cumulative=-1)
    #plt.subplot(212), plt.title('Histogram Bounding Box Areas')
    #plt.subplot(212), plt.xlabel('μm')
    #plt.subplot(212), plt.ylabel('Occurence')
    #plt.subplot(212), plt.grid(True)

    #plt.show()

    #listPerimeters
    #plt.subplot(311), plt.hist(listECPDiameter, bins=100, density=True, histtype='step', cumulative=True)
    #plt.subplot(311), plt.hist(listECPDiameter, bins=100, density=True, histtype='step', cumulative=-1)
    #plt.subplot(311), plt.title('Histogram Perimeter')
    #plt.subplot(311), plt.xlabel('μm')
    #plt.subplot(311), plt.ylabel('Occurence')
    #plt.subplot(311), plt.grid(True)

    #listCPM
    #plt.subplot(312), plt.hist(listCPM, bins=100, density=True, histtype='step', cumulative=True)
    #plt.subplot(312), plt.hist(listCPM, bins=100, density=True, histtype='step', cumulative=-1)
    #plt.subplot(312), plt.title('Histogram Convex Perimeter')
    #plt.subplot(312), plt.xlabel('μm')
    #plt.subplot(312), plt.ylabel('Occurence')
    #plt.subplot(312), plt.grid(True)
    
    #listECPDiameter
    #plt.subplot(313), plt.hist(listECPDiameter, bins=100, density=True, histtype='step', cumulative=True)
    #plt.subplot(313), plt.hist(listECPDiameter, bins=100, density=True, histtype='step', cumulative=-1)
    #plt.subplot(313), plt.title('Histogram Equivalent Circular Perimeter Diameter')
    #plt.subplot(313), plt.xlabel('μm')
    #plt.subplot(313), plt.ylabel('Occurence')
    #plt.subplot(313), plt.grid(True)

    #plt.show()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot(dap, db)
    num = dot(dap, dp)
    return (num / denom.astype(float))*db + b1

def feretsDiameters(hull, step=1):
    #step = 2 * (math.pi/180)
    hull = np.array(hull)
    theta = np.degrees(step)
    c, s = np.cos(theta), np.sin(theta)
    rotationMatrix = np.array(((c, -s), (s, c)))
    maxDiameter = -math.inf
    minDiameter = math.inf
    #print("Hull: " + str(hull))
    #print("RotationMatrix: " + str(rotationMatrix))

    for angle in range(0, 90, step):
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rotationMatrix = np.array(((c, -s), (s, c)))

        for x, point in enumerate(hull):
            hull[x] = np.dot(hull[x], rotationMatrix)

        _, _, w, h = cv2.boundingRect(hull)
        if minDiameter > w:
            minDiameter = w
        if minDiameter > h:
            minDiameter = h
        if maxDiameter < w:
            maxDiameter = w
        if maxDiameter < h:
            maxDiameter = h

    return (minDiameter, maxDiameter)

def students_ttest_wilcoxon(inputMeasures, functionName):
    low = 2 * MICRO_METER_PER_PIXEL
    high = 3500
    
    inputMeasures = [i for j, i in enumerate(inputMeasures) if j > DIAMETER_THRESHOLD_LOW or j < DIAMETER_THRESHOLD_HIGH]
    listMeasureX0 = list()
    
    for percentile in CDF_PERCENTILE_REF:
        listMeasureX0.append(np.percentile(inputMeasures, percentile))
    tmp = np.array((listMeasureX0, CDF_PERCENTILE_REF))
    tmpRef = np.array((CDF_X0_REF, CDF_PERCENTILE_REF))

    #print("Standard Deviation Ref: " + str(tmpRef.std()))
    #print("Standard Deviation x0: " + str(tmp.std()))
    
    (studentsStat, studentsPval) = scipy.stats.ttest_ind(listMeasureX0, CDF_X0_REF, equal_var=True)
    #(wilcoxonStat, wilcoxonPval) = scipy.stats.wilcoxon(listMeasureX0, CDF_X0_REF)
    print(str(functionName) + " " + "Student's t-test: " + str((studentsStat, studentsPval)))
    #print(str(functionName) + " " + "Wilcoxon signed-rank test " + str((wilcoxonStat, wilcoxonPval)))

    if (DEBUG_DATA):
        plt.plot(CDF_X0_REF, CDF_PERCENTILE_REF, 'o')
        plt.plot(listMeasureX0, CDF_PERCENTILE_REF)
        plt.grid(True)
        plt.xscale('log')
        plt.xlim([10, 4000])
        plt.ylim([0, 102])
        plt.show()

    return (studentsStat, studentsPval, 0, 0)#, wilcoxonStat, wilcoxonPval)

def sphereVolume(diameter):
    #return (4/3) * math.pi * (diameter/2)**3
    return 4 * math.pi * (diameter/2)**2

def applyWeight(listsToWeight):
    for idx, l in enumerate(listsToWeight):
        (hist, other) = np.histogram(l, bins=1000000, weights=[sphereVolume(d) for d in l], density=True, normed=True)
        #print("Hist: " + str(hist))
        #print("Other: " + str(len(other)))
        listsToWeight[idx] = hist
    return listsToWeight
















##### MAIN ####
def main():
    global DEBUG_FILTERING, DEBUG_SEGMENTING, DEBUG_EDITING, DEBUG_PLOT, DEBUG_DATA, DEBUG_STUDENTS_TTEST, DEBUG_SAVE, FILENAME, X50REF, X503TEMP, CDF_PERCENTILE_REF, EXCEL_ROW, DEBUG_HEIGHTS, DEBUG_WIDHTS, DEBUG_LFD, DEBUG_GFD, DEBUG_MFD, DEBUG_ECPD, DEBUG_ECAD, DEBUG_LBCD, DEBUG_HMD, DEBUG_VMD, DEBUG_LBRW, DEBUG_LBRL, DEBUG_FL, DEBUG_FW, DEBUG_MAJOR_AXIS, DEBUG_MINOR_AXIS

    DEBUG_FILTERING = False
    DEBUG_SEGMENTING = False
    DEBUG_EDITING = False
    DEBUG_PLOT = False
    DEBUG_DATA = False
    DEBUG_STUDENTS_TTEST = True
    DEBUG_TEST_CAL = False
    DEBUG_TEST_ALL = False
    DEBUG_SAVE = False

    DEBUG_HEIGHTS = False
    DEBUG_WIDHTS = False
    DEBUG_LFD = False
    DEBUG_GFD = False
    DEBUG_MFD = False
    DEBUG_ECPD = False
    DEBUG_ECAD = False
    DEBUG_LBCD = False
    DEBUG_HMD = False
    DEBUG_VMD = False
    DEBUG_LBRW = False
    DEBUG_LBRL = False
    DEBUG_FL = False
    DEBUG_FW = False
    DEBUG_MAJOR_AXIS = False
    DEBUG_MINOR_AXIS = False

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["debug", "plot", "data", "filename=", "x50=", "test-cal", "test-all", "save",
                                                    "heights",
                                                    "widths",
                                                    "LFD",
                                                    "GFD",
                                                    "MFD",
                                                    "ECPD",
                                                    "ECAD",
                                                    "LBCD",
                                                    "HMD",
                                                    "VMD",
                                                    "LBRW",
                                                    "LBRL",
                                                    "FL",
                                                    "FW",
                                                    "majorAxis",
                                                    "minorAxis"])

    except getopt.GetoptError as err:
        print(err)
        print("error")
        sys.exit(2)

    # process options
    #for o, a in opts:
    #    print("o: " + str(o) + "\na: " + str(a))
    for o, a in opts:
        if o in ("--debug"):
            print("--debug")
            DEBUG_FILTERING = True
            DEBUG_SEGMENTING = True
            DEBUG_EDITING = True
        elif o in ("--plot"):
            print("--plot")
            DEBUG_PLOT = True
        elif o in ("--data"):
            print("--data")
            DEBUG_DATA = True
        elif o in ("--filename="):
            FILENAME = "Images/" + a + "/latest0.jpg"
        elif o in ("--x50="):
            X50REF = a
        elif o in ("--test-cal"):
            DEBUG_TEST_CAL = True
        elif o in ("--test-all"):
            DEBUG_TEST_ALL = True
        elif o in ("--save"):
            DEBUG_SAVE = True
        elif o in ("--heights"):
            DEBUG_HEIGHTS = True
        elif o in ("--widths"):
            DEBUG_WIDHTS = True
        elif o in ("--LFD"):
            DEBUG_LFD = True
        elif o in ("--GFD"):
            DEBUG_GFD = True
        elif o in ("--MFD"):
            DEBUG_MFD = True
        elif o in ("--ECPD"):
            DEBUG_ECPD = True
        elif o in ("--ECAD"):
            DEBUG_ECAD = True
        elif o in ("--LBCD"):
            DEBUG_LBCD = True
        elif o in ("--HMD"):
            DEBUG_HMD = True
        elif o in ("--VMD"):
            DEBUG_VMD = True
        elif o in ("--LBRW"):
            DEBUG_LBRW = True
        elif o in ("--LBRL"):
            DEBUG_LBRL = True
        elif o in ("--FL"):
            DEBUG_FL = True
        elif o in ("--FW"):
            DEBUG_FW = True
        elif o in ("--majorAxis"):
            DEBUG_MAJOR_AXIS = True
        elif o in ("--minorAxis"):
            DEBUG_MINOR_AXIS = True

    # process arguments
    for arg in args:
        process(arg) # process() is defined elsewhere

    if (DEBUG_TEST_ALL or DEBUG_TEST_CAL):
        listTest = list()

        if (DEBUG_TEST_CAL):
            listTest.append((11, 470, 470, []))
            listTest.append((12, 470, 470, []))
            listTest.append((13, 470, 470, []))
            listTest.append((14, 470, 470, []))
            listTest.append((21, 470, 470, []))
            listTest.append((22, 470, 470, []))
            listTest.append((23, 470, 470, []))
            listTest.append((24, 470, 470, []))
            listTest.append((31, 470, 470, []))
            listTest.append((32, 470, 470, []))
            listTest.append((33, 470, 470, []))
            listTest.append((34, 470, 470, []))
            listTest.append((41, 470, 470, []))
            listTest.append((42, 470, 470, []))
            listTest.append((43, 470, 470, []))
            listTest.append((44, 470, 470, []))
            listTest.append((51, 470, 470, []))
            listTest.append((52, 470, 470, []))
            listTest.append((53, 470, 470, []))
            listTest.append((54, 470, 470, []))

        if (DEBUG_TEST_ALL):
            listTest.append((1040, 394, 441, [7.58,  8.91,  10.14, 11.27, 12.83, 14.65, 16.18, 17.72,
                                              19.13, 20.18, 21.15, 22.31, 23.98, 25.72, 27.58, 30.33,
                                              34.31, 40.12, 47.19, 58.01, 71.45, 84.65, 94.52, 100.0,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1084, 396, 452, [6.97,  8.30,  9.53,  10.67, 12.19, 13.90, 15.30, 16.70,
                                              17.99, 18.99, 19.94, 21.09, 22.70, 24.38, 26.12, 28.94,
                                              33.01, 38.90, 45.85, 56.15, 68.74, 80.92, 90.13, 95.66,
                                              98.63, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1108, 449, 486, [6.78,  8.06,  9.24,  10.31, 11.74, 13.34, 15.63, 15.92,
                                              17.09, 18.00, 18.86, 19.90, 21.38, 22.91, 24.57, 27.05,
                                              30.70, 35.99, 42.28, 51.69, 63.38, 74.99, 84.09, 89.86,
                                              93.44, 95.77, 97.76, 99.36, 100.0, 100.0, 100.0]))
            listTest.append((1109, 401, 417, [7.45,  8.92,  10.29, 11.54, 13.22, 15.13, 16.70, 18.28,
                                              19.72, 20.83, 21.87, 23.14, 24.91, 26.77, 28.79, 31.83,
                                              36.38, 42.92, 50.32, 60.66, 72.55, 83.40, 91.37, 96.38,
                                              99.21, 99.97, 100.0, 100.0, 100.0, 100.0, 100.0]))
            #TODO listTest.append((1150, 426, 416, []))
            listTest.append((1156, 422, 435, [7.16,  8.48,  9.69,  10.81, 12.33, 14.07, 15.53, 17.01,
                                              18.38, 19.46, 20.46, 21.67, 23.34, 25.09, 26.99, 29.85,
                                              34.10, 40.40, 47.87, 59.08, 73.17, 86.77, 95.83, 99.32,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1159, 409, 483, [6.17,  7.38,  8.50,  9.53,  10.93, 12.53, 13.86, 15.19,
                                              16.41, 17.35, 18.23, 19.31, 20.83, 22.42, 24.11, 26.60,
                                              30.25, 35.59, 42.06, 52.07, 65.21, 79.24, 90.73, 97.26,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1162, 388, 457, [6.84,  8.17,  9.41,  10.53, 12.04, 13.74, 15.14, 16.54,
                                              17.83, 18.82, 19.76, 20.90, 22.50, 24.16, 25.95, 28.64,
                                              32.60, 38.38, 45.25, 55.48, 68.12, 80.73, 90.68, 96.62,
                                              99.24, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1168, 426, 542, [5.98,  7.02,  8.07,  9.03,  10.32, 11.78, 12.98, 14.17,
                                              15.24, 16.07, 16.84, 17.78, 19.10, 20.49, 21.97, 24.13,
                                              27.27, 31.81, 37.23, 45.50, 56.20, 67.41, 76.93, 83.84,
                                              89.19, 93.72, 97.66, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1175, 428, 361, [8.57,  10.28, 11.85, 13.29, 15.23, 17.44, 19.26, 21.08,
                                              22.74, 24.01, 25.19, 26.64, 28.71, 30.90, 33.29, 36.93,
                                              42.28, 49.80, 58.12, 69.32, 81.27, 91.11, 97.17, 100.0,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1177, 421, 537, [5.72,  6.74,  7.68,  8.54,  9.72,  11.08, 12.22, 13.39,
                                              14.48, 15.32, 16.11, 17.07, 18.43, 19.83, 21.30, 23.45,
                                              26.53, 30.96, 36.41, 45.28, 57.97, 73.04, 86.81, 95.65,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1181, 439, 525, [5.51,  6.59,  7.60,  8.53,  9.79,  11.24, 12.43, 13.63,
                                              14.72, 15.56, 16.35, 17.32, 18.70, 20.15, 21.67, 23.92,
                                              27.21, 32.00, 37.82, 46.94, 59.23, 72.67, 84.37, 92.54,
                                              97.56, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1187, 390, 389, [8.75,  10.48, 12.07, 13.51, 15.45, 17.62, 19.38, 21.11,
                                              22.64, 23.77, 24.81, 26.07, 27.88, 29.80, 31.87, 34.99,
                                              39.63, 46.30, 53.90, 64.59, 76.87, 87.90, 95.33, 98.80,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1190, 406, 428, [7.22,  8.64,  9.95,  11.15, 12.77, 14.60, 16.11, 17.61,
                                              18.98, 20.02, 21.00, 22.20, 23.92, 25.74, 27.71, 30.69,
                                              35.10, 41.51, 48.93, 59.57, 72.05, 83.55, 91.87, 96.72,
                                              99.14, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1191, 409, 517, [5.27,  6.31,  7.28,  8.18,  9.41,  10.83, 12.01, 13.21,
                                              14.33, 15.21, 16.03, 17.06, 18.50, 20.00, 21.59, 23.92,
                                              27.33, 32.28, 38.33, 47.83, 60.47, 74.11, 85.83, 93.70,
                                              98.13, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1197, 409, 532, [5.79,  6.83,  7.79,  8.68,  9.88,  11.27, 12.43, 13.61,
                                              14.69, 15.53, 16.32, 17.28, 18.65, 20.07, 21.57, 23.77,
                                              26.91, 31.43, 36.96, 45.92, 58.70, 73.68, 87.10, 95.67,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1206, 421, 522, [5.92,  6.99,  7.98,  8.88,  10.10, 11.49, 12.65, 13.82,
                                              14.92, 15.78, 16.60, 17.58, 18.94, 20.35, 21.85, 24.04,
                                              27.22, 31.91, 37.75, 47.17, 60.20, 74.96, 87.91, 96.02,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1207, 356, 459, [7.64,  9.13,  10.50, 11.74, 13.39, 15.25, 16.76, 18.25,
                                              19.57, 20.57, 21.48, 22.59, 24.16, 25.79, 27.52, 30.05,
                                              33.71, 38.97, 45.31, 55.02, 67.54, 80.64, 91.24, 97.30,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1210, 401, 506, [5.69,  6.80,  7.83,  8.78,  10.07, 11.55, 12.79, 14.04,
                                              15.19, 16.08, 16.92, 17.93, 19.35, 20.82, 22.38, 24.71,
                                              28.14, 33.22, 39.44, 49.20, 62.22, 76.28, 88.14, 95.76,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1213, 394, 518, [5.77,  6.88,  7.92,  8.88,  10.16, 11.63, 12.84, 14.06,
                                              15.18, 16.04, 16.85, 17.84, 19.23, 20.67, 22.20, 24.45,
                                              27.74, 32.59, 38.48, 47.72, 60.12, 73.59, 85.16, 93.12,
                                              97.85, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1214, 431, 437, [7.30,  8,73,  10.05, 11.26, 12.88, 14.71, 16.21, 17.69,
                                              19.02, 20.02, 20.95, 22.09, 23.72, 25.45, 27.34, 30.18,
                                              34.36, 40.44, 47.73, 58.66, 72.08, 84.89, 93.98, 98.41,
                                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1217, 437, 466, [6.74,  8.06,  9.28,  10.40, 11.90, 13.61, 15.01, 16.41,
                                              17.69, 18.66, 19.58, 20.70, 22.27, 23.90, 25.65, 28.24,
                                              32.04, 37.58, 44.21, 54.25, 66.86, 79.54, 89.61, 95.88,
                                              98.93, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11095, 450, 383, [8.36,  9.91,  11.34, 12.66, 14.43, 16.43, 18.09, 19.75,
                                               21.29, 22.49, 23.63, 25.00, 26.91, 28.92, 31.13, 34.50,
                                               39.51, 46.78, 55.11, 66.78, 79.65, 90.54, 97.20, 99.58,
                                               100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11096, 414, 477, [6.93,  8.28,  9.53,  10.67, 12.19, 13.89, 15.26, 16.61,
                                               17.83, 18.75, 19.63, 20.69, 22.19, 23.75, 25.43, 27.91,
                                               31.55, 36.84, 43.19, 52.82, 64.97, 77.19, 86.88, 93.10,
                                               96.84, 98.79, 99.74, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11104, 416, 456, [6.66,  7.97,  9.17,  10.28, 11.78, 13.48, 14.88, 16.29,
                                               17.58, 18.58, 19.52, 20.66, 22.26, 23.91, 25.67, 28.33,
                                               32.32, 38.18, 45.24, 55.82, 68.74, 81.48, 91.53, 97.39,
                                               99.63, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11112, 387, 476, [5.99,  7.14,  8.22,  9.20,  10.53, 12.05, 13.32, 14.59,
                                               15.78, 16.70, 17.58, 18.66, 20.18, 21.78, 23.50, 26.08,
                                               29.90, 35.66, 42.63, 53.20, 66.44, 79.60, 89.89, 96.12,
                                               99.09, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11133, 391, 472, [6.42,  7.59,  8.66,  9.64,  10.98, 12.51, 13.79, 15.10,
                                               16.32, 17.27, 18.17, 19.26, 20.77, 22.34, 24.04, 26.60,
                                               30.39, 36.07, 43.01, 53.81, 67.88, 82.39, 93.31, 98.47,
                                               100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            #TODO listTest.append((11144, 410, 478, []))
            listTest.append((11146, 420, 539, [5.53,  6.61,  7.61,  8.54,  9.80,  11.23, 12.41, 13.61,
                                               14.71, 15.56, 16.37, 17.34, 18.70, 20.09, 21.55, 23.67,
                                               26.70, 31.04, 36.40, 45.12, 57.51, 72.03, 85.37, 94.67,
                                               100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            #TODO listTest.append((11147, 405, 579, []))
            listTest.append((11158, 416, 509, [5.81,  6.93,  7.97,  8.93,  10.22, 11.69, 12.90, 14.12,
                                               15.23, 16.09, 16.91, 17.91, 19.32, 20.80, 22.41, 24.80,
                                               28.30, 33.36, 39.43, 48.84, 61.30, 74.75, 86.28, 94.21,
                                               98.73, 99.96, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11160, 419, 507, [5.94,  7.08,  8.14,  9.11,  10.42, 11.91, 13.14, 14.38,
                                               15.51, 16.38, 17.20, 18.20, 19.60, 21.06, 22.62, 24.92,
                                               28.30, 33.32, 39.46, 49.09, 62.03, 76.17, 88.19, 95.88,
                                               100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11199, 434, 465, [5.79,  6.94,  8.02,  9.03,  10.41, 12.01, 13.36, 14.73,
                                               16.00, 16.99, 17.92, 19.06, 20.68, 22.35, 24.16, 26.86,
                                               30.88, 36.83, 43.94, 54.67, 68.13, 81.13, 91.42, 97.10,
                                               100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11204, 407, 545, [6.62,  7.82,  8.92,  9.93,  11.26, 12.75, 13.96, 15.15,
                                               16.25, 17.10, 17.90, 18.87, 20.22, 21.60, 23.05, 25.13,
                                               28.03, 32.12, 37.06, 44.99, 56.20, 69.49, 82.20, 91.66,
                                               97.36, 99.46, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11216, 405, 443, [8.02,  9.47,  10.79, 12.00, 13.62, 15.45, 16.95, 18.43,
                                               19.78, 20.81, 21.77, 22.92, 24.54, 26.23, 28.05, 30.74,
                                               34.66, 40.34, 47.05, 57.20, 70.30, 83.49, 93.22, 98.14,
                                               100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((11222, 415, 453, [6.75,  8.09,  9.34,  10.50, 12.05, 13.83, 15.28, 16.73,
                                               18.04, 19.03, 19.96, 21.09, 22.70, 24.38, 26.20, 28.91,
                                               32.98, 38.84, 45.75, 56.03, 68.89, 81.85, 91.90, 97.60,
                                               99.69, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))

        EXCEL_ROW = 0
        for (file, temp3, expected, cdfPercentiles) in listTest:
            print("Arguments")
            print("Filename: " + str(file))
            print("X503TEMP: " + str(temp3))
            print("X50REF  : " + str(expected))
            print("cdfPercentiles: " + str(cdfPercentiles))
            FILENAME = "images/" + str(file) + "/latest0.jpg"
            X503TEMP = temp3
            X50REF = expected
            if (cdfPercentiles != []):
                CDF_PERCENTILE_REF = cdfPercentiles
            else:
                CDF_PERCENTILE_REF = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                      100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                      100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                      100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

            EXCEL_ROW += 1
            analyze()
            print("")
    else:
        analyze()

    wb.close()

def process(arg):
    print(arg)

if __name__ == "__main__":
    main()


# wilcox's ranksum test
# kruskal-wallis test by ranks h-test