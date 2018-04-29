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
global FILENAME
FILENAME = 'images/image/latest0.jpg'
global X50REF
global X503TEMP
global CDF_PERCENTILE_REF
X50REF = 1
X503TEMP = 1
CDF_PERCENTILE_REF = []
CDF_X0_REF = [18, 22, 26, 30, 36, 44, 52, 62, # TODO Histogram och standard variation
            74, 86, 100, 120, 150, 180, 210, 250,
            300, 360, 420, 500, 600, 720, 860, 1020,
            1220, 1460, 1740, 2060, 2460, 2940, 3500]
# TODO um / pixel microMeterPerPixel = 20.0 # 19.8
MICRO_METER_PER_PIXEL = 17.4 #18.15 * 2.0#19.8 #17.39
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
excel_sheet.write(1, 0, float(1.0))
excel_sheet.write(0, 1, 'Filename')
excel_sheet.write(0, 2, 'Reference x50')
excel_sheet.write(0, 3, '3TEMP x50')
excel_sheet.write(0, 4, '3TEMP Deviation')
excel_sheet.write(0, 5, 'Heights x50')
excel_sheet.write(0, 6, 'Deviation')
excel_sheet.write(0, 7, 'Standard Deviation')
excel_sheet.write(0, 8, 'Variance')
excel_sheet.write(0, 9, 'Widths x50')
excel_sheet.write(0, 10, 'Deviation')
excel_sheet.write(0, 11, 'Standard Deviation')
excel_sheet.write(0, 12, 'Variance')
excel_sheet.write(0, 13, 'Least Feret Diameter x50')
excel_sheet.write(0, 14, 'Deviation')
excel_sheet.write(0, 15, 'Standard Deviation')
excel_sheet.write(0, 16, 'Variance')
excel_sheet.write(0, 17, 'Greatest Feret Diameter x50')
excel_sheet.write(0, 18, 'Deviation')
excel_sheet.write(0, 19, 'Standard Deviation')
excel_sheet.write(0, 20, 'Variance')
excel_sheet.write(0, 21, 'Equivalent Circle Perimeter Diameter x50')
excel_sheet.write(0, 22, 'Deviation')
excel_sheet.write(0, 23, 'Standard Deviation')
excel_sheet.write(0, 24, 'Variance')
excel_sheet.write(0, 25, 'Equivalent Circle Area Diameter x50')
excel_sheet.write(0, 26, 'Deviation')
excel_sheet.write(0, 27, 'Standard Deviation')
excel_sheet.write(0, 28, 'Variance')
excel_sheet.write(0, 29, 'Least Bounding Circle Diameter x50')
excel_sheet.write(0, 30, 'Deviation')
excel_sheet.write(0, 31, 'Standard Deviation')
excel_sheet.write(0, 32, 'Variance')
excel_sheet.write(0, 33, 'Horizontal Martin Diameter x50')
excel_sheet.write(0, 34, 'Deviation')
excel_sheet.write(0, 35, 'Standard Deviation')
excel_sheet.write(0, 36, 'Variance')
excel_sheet.write(0, 37, 'Vertical Martin Diameter x50')
excel_sheet.write(0, 38, 'Deviation')
excel_sheet.write(0, 39, 'Standard Deviation')
excel_sheet.write(0, 40, 'Variance')
excel_sheet.write(0, 41, 'Least Bounding Rectagle Width x50')
excel_sheet.write(0, 42, 'Deviation')
excel_sheet.write(0, 43, 'Standard Deviation')
excel_sheet.write(0, 44, 'Variance')
excel_sheet.write(0, 45, 'Least Bounding Rectangle Length x50')
excel_sheet.write(0, 46, 'Deviation')
excel_sheet.write(0, 47, 'Standard Deviation')
excel_sheet.write(0, 48, 'Variance')
excel_sheet.write(0, 49, 'Fiber Length x50')
excel_sheet.write(0, 50, 'Deviation')
excel_sheet.write(0, 51, 'Standard Deviation')
excel_sheet.write(0, 52, 'Variance')
excel_sheet.write(0, 53, 'Fiber Width x50')
excel_sheet.write(0, 54, 'Deviation')
excel_sheet.write(0, 55, 'Standard Deviation')
excel_sheet.write(0, 56, 'Variance')
excel_sheet.write(0, 57, 'Major Axis Length x50')
excel_sheet.write(0, 58, 'Deviation')
excel_sheet.write(0, 59, 'Standard Deviation')
excel_sheet.write(0, 60, 'Variance')
excel_sheet.write(0, 61, 'Minor Axis Length x50')
excel_sheet.write(0, 62, 'Deviation')
excel_sheet.write(0, 63, 'Standard Deviation')
excel_sheet.write(0, 64, 'Variance')

global DEBUG_FILTERING
global DEBUG_SEGMENTING
global DEBUG_EDITING
global DEBUG_PLOT
global DEBUG_DATA
global DEBUG_STUDENTS_TTEST

HYBRID_FILTER = True
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
    global listGrainImages, listHeights, listWidths, listLFDiameter, listGFDiameter, listBoundingBoxAreas, listAreas, listCPM, listECPDiameter, listPerimeters, listCentroids, listECADiameter, listLBCDiameter, listHMDiameter, listVMDiameter, listLBRW, listLBRL, listFiberLength, listFiberWidth, listFittedEllipse, listMajorAxisLength, listMinorAxisLength, listIsConvex
 
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

            # Horizontal Feret’s Diameter (HFD): The distance between two parallel lines at horizontal direction that do not intersect the image.
            # Width of boundingbox
            width = (boundingBoxRowMax - boundingBoxRowMin) * MICRO_METER_PER_PIXEL
            listWidths.append(width)

            # Vertical Feret’s Diameter (VFD): The distance between two parallel lines at vertical direction that do not intersect the image.
            # Height of boundingbox
            height = (boundingBoxColMax - boundingBoxColMin) * MICRO_METER_PER_PIXEL
            listHeights.append(height)

            # Area
            listAreas.append(props.area * MICRO_METER_PER_PIXEL * MICRO_METER_PER_PIXEL)
            listBoundingBoxAreas.append(width * height)

            # Least Feret’s Diameter (LFD), Feret’s Width: The smallest distance between two parallel lines that do not intersect the image.
            # Greatest Feret’s Diameter (GFD), Feret’s Length: The greatest distance between two parallel lines that do not intersect the image.
            (LFDiameter, GFDiameter) = feretsDiameters(contour)
            listLFDiameter.append(LFDiameter * MICRO_METER_PER_PIXEL)
            listGFDiameter.append(GFDiameter * MICRO_METER_PER_PIXEL)

            # Equivalent Circular Area Diameter (ECAD), Heywood’s Diameter: The diameter of a circle that has the same area as the image.
            diameter = math.sqrt(cv2.contourArea(contour) / math.pi) * 2 * MICRO_METER_PER_PIXEL
            listECADiameter.append(diameter)
            if diameter < DIAMETER_THRESHOLD_LOW * 2 or diameter > DIAMETER_THRESHOLD_HIGH:
                count += 1

            # Least Bounding Circle (LBC): The smallest circle that encloses the image.
            (x0, y0), radius = cv2.minEnclosingCircle(contour)
            center = (int(x0), int(y0))
            #radius = int(radius)
            #img = cv2.circle(img, center, radius, (0,255,0), 2)
            listLBCDiameter.append(radius * 2 * MICRO_METER_PER_PIXEL)

            # Convex Perimeter (CPM): The perimeter of a convex curve circumscribing the image.
            convex_contour = cv2.convexHull(contour)
            listCPM.append(cv2.arcLength(convex_contour, True) * MICRO_METER_PER_PIXEL)

            # Equivalent Circular Perimeter Diameter (ECPD): The diameter of a circle that has the same perimeter of the image.
            imagePerimeter = cv2.arcLength(contour, True)
            #imagePerimeter = 2 * (boundingBoxRowMax - boundingBoxRowMin) + 2 * (boundingBoxColMax - boundingBoxColMin) * MICRO_METER_PER_PIXEL
            radius = imagePerimeter / (2 * math.pi)
            listECPDiameter.append(radius * 2 * MICRO_METER_PER_PIXEL)

            # Horizontal Martin’s Diameter (HMD): The length of a line at horizontal direction that divides the image into two equal halves.
            # Vertical Martin’s Diameter (VMD): The length of a line at vertical direction that divides the image into two equal halves.
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

            (x, y), (w, h), r = cv2.minAreaRect(contour)
            # Least Bounding Rectangle Width (LBRW): The width of the smallest rectangle that encloses the image.
            # Least Bounding Rectangle Length (LBRL): The length of the smallest rectangle that encloses the image.
            listLBRW.append(w * MICRO_METER_PER_PIXEL)
            listLBRL.append(h * MICRO_METER_PER_PIXEL)

            # Fiber Length (FL): The length of a rectangle that has the same area and perimeter as the image.
            # Fiber Width (FW): The width of a rectangle that has the same area and perimeter as the image.
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
            ax.plot(x0+1, y0+1, '.r', markersize=5)
            ax.add_artist(plt.Circle((x0+1, y0+1), listECADiameter[idx]/2.0, color='r', fill=False, linestyle='dashed'))

            plt.show()

    # Write all x50 to excel
    listHeights.sort()
    listWidths.sort()
    listLFDiameter.sort()
    listGFDiameter.sort()
    listECPDiameter.sort()
    listECADiameter.sort()
    listLBCDiameter.sort()
    listHMDiameter.sort()
    listVMDiameter.sort()
    listLBRW.sort()
    listLBRL.sort()
    listFiberLength.sort()
    listFiberWidth.sort()
    listMajorAxisLength.sort()
    listMinorAxisLength.sort()

    arrayListHeights = np.array(listHeights)
    arrayListWidths = np.array(listWidths)
    arrayListLFDiameter = np.array(listLFDiameter)
    arrayListGFDiameter = np.array(listGFDiameter)
    arrayListECPDiameter = np.array(listECPDiameter)
    arrayListECADiameter = np.array(listECADiameter)
    arrayListLBCDiameter = np.array(listLBCDiameter)
    arrayListHMDiameter = np.array(listHMDiameter)
    arrayListVMDiameter = np.array(listVMDiameter)
    arrayListLBRW = np.array(listLBRW)
    arrayListLBRL = np.array(listLBRL)
    arrayListFiberLength = np.array(listFiberLength)
    arrayListFiberWidth = np.array(listFiberWidth)
    arrayListMajorAxisLength = np.array(listMajorAxisLength)
    arrayListMinorAxisLength = np.array(listMinorAxisLength)

    listMeasures = list()
    listMeasures.append(arrayListHeights)
    listDebugMeasurementTitles.append("Height")
    listMeasures.append(arrayListWidths)
    listDebugMeasurementTitles.append("Width")
    listMeasures.append(arrayListLFDiameter)
    listDebugMeasurementTitles.append("Least Feret Diameter")
    listMeasures.append(arrayListGFDiameter)
    listDebugMeasurementTitles.append("Greatest Feret Diameter")
    listMeasures.append(arrayListECPDiameter)
    listDebugMeasurementTitles.append("Equivalent Circle Perimeter Diameter")
    listMeasures.append(arrayListECADiameter)
    listDebugMeasurementTitles.append("Equivalent Circle Area Diameter")
    listMeasures.append(arrayListLBCDiameter)
    listDebugMeasurementTitles.append("Least Bounding Circle Diameter")
    listMeasures.append(arrayListHMDiameter)
    listDebugMeasurementTitles.append("Horizontal Martin Diameter")
    listMeasures.append(arrayListVMDiameter)
    listDebugMeasurementTitles.append("Vertical Martin Diameter")
    listMeasures.append(arrayListLBRW)
    listDebugMeasurementTitles.append("Least Bounding Rectangle Width")
    listMeasures.append(arrayListLBRL)
    listDebugMeasurementTitles.append("Least Bounding Rectangle Height")
    listMeasures.append(arrayListFiberLength)
    listDebugMeasurementTitles.append("Fiber Length")
    listMeasures.append(arrayListFiberWidth)
    listDebugMeasurementTitles.append("Fiber Width")
    listMeasures.append(arrayListMajorAxisLength)
    listDebugMeasurementTitles.append("Major Axis")
    listMeasures.append(arrayListMinorAxisLength)
    listDebugMeasurementTitles.append("Minor Axis")

    x50Heights = np.percentile(arrayListHeights, 50)
    x50Widths = np.percentile(arrayListWidths, 50)
    x50LFDiameter = np.percentile(arrayListLFDiameter, 50)
    x50GFDiameter = np.percentile(arrayListGFDiameter, 50)
    x50ECPDiameter = np.percentile(arrayListECPDiameter, 50)
    x50ECADiameter = np.percentile(arrayListECADiameter, 50)
    x50LBCDiameter = np.percentile(arrayListLBCDiameter, 50)
    x50HMDiameter = np.percentile(arrayListHMDiameter, 50)
    x50VMDiameter = np.percentile(arrayListVMDiameter, 50)
    x50LBRW = np.percentile(arrayListLBRW, 50)   
    x50LBRL = np.percentile(arrayListLBRL, 50)    
    x50FiberLength = np.percentile(arrayListFiberLength, 50)   
    x50FiberWidth = np.percentile(arrayListFiberWidth, 50)
    x50MajorAxisLength = np.percentile(arrayListMajorAxisLength, 50)
    x50MinorAxisLength = np.percentile(arrayListMinorAxisLength, 50)

    listX50Measures = list()
    listX50Measures.append(x50Heights)
    listX50Measures.append(x50Widths)
    listX50Measures.append(x50LFDiameter )
    listX50Measures.append(x50GFDiameter )
    listX50Measures.append(x50ECPDiameter)
    listX50Measures.append(x50ECADiameter)
    listX50Measures.append(x50LBCDiameter)
    listX50Measures.append(x50HMDiameter)
    listX50Measures.append(x50VMDiameter)
    listX50Measures.append(x50LBRW)
    listX50Measures.append(x50LBRL)
    listX50Measures.append(x50FiberLength)
    listX50Measures.append(x50FiberWidth)
    listX50Measures.append(x50MajorAxisLength)
    listX50Measures.append(x50MinorAxisLength)

    temp3Deviation = X503TEMP/float(X50REF)*100 - 100

    excel_sheet.write(EXCEL_ROW, 1, str(FILENAME))
    excel_sheet.write(EXCEL_ROW, 2, str(X50REF))
    excel_sheet.write(EXCEL_ROW, 3, str(X503TEMP))
    if math.fabs(temp3Deviation) < 10:
        excel_sheet.write(EXCEL_ROW, 4, temp3Deviation, formatGreen)
    else:
        excel_sheet.write(EXCEL_ROW, 4, temp3Deviation, formatRed)

    for idx, measure in enumerate(listMeasures):
        #measureDeviation = measure/float(X50REF)*100 - 100
        x50measure = np.percentile(measure, 50)
        NUMBER_OF_DATA_ENTRIES = 4
        NUMBER_OF_FIXED_ENTRIES = 5

        excel_sheet.write(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES, x50measure) # TODO
        excel_sheet.write(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+1, str('=ABS($A$2*OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())),0,-1)/OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, ' + str(-4-idx*NUMBER_OF_DATA_ENTRIES) + ')*100-100)'))
        excel_sheet.write(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+2, str(statistics.pstdev(measure)))
        excel_sheet.write(EXCEL_ROW, idx*NUMBER_OF_DATA_ENTRIES+NUMBER_OF_FIXED_ENTRIES+3, str(statistics.pvariance(measure)))

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

    if (DEBUG_STUDENTS_TTEST):
        students_ttest(listMeasures)

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
    print("Number of dirt grains:" + str(count))
    print("Number of dirt grains %:" + str(100*count/len(regions)))

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

    print("Standard Deviation: " + str(statistics.pstdev(listFiberLength)))
    print("Variance: " + str(statistics.pvariance(listFiberLength)))
    #print("Q1: " + str(Q1))
    #print("Q2: " + str(Q2))
    #print("Q3: " + str(Q3))

    if (DEBUG_PLOT):
        plotImages()

    if (DEBUG_DATA):
        plotData()

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
    image = inputImage <= 50#threshold #TODO Får inte samma värde som Anders, belysningen räknas in här
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
    listDebugImageTitles.append('Distance')

    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
    listDebugImages.append(local_maxi)
    listDebugImageTitles.append('Local Maxi')

    markers = ndi.label(local_maxi)[0]

    labels = watershed(-distance, markers, mask=image)
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

    # Compact Data Plot
    listHeights.sort()
    listWidths.sort()
    listLFDiameter.sort()
    listGFDiameter.sort()
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
    plt.hist(listHeights, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listWidths, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listLFDiameter, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listGFDiameter, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listECPDiameter, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listLBCDiameter, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listHMDiameter, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listVMDiameter, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listLBRW, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listLBRL, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listFiberLength, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listFiberWidth, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listMajorAxisLength, bins=100000, density=True, histtype='step', cumulative=True)
    plt.hist(listMinorAxisLength, bins=100000, density=True, histtype='step', cumulative=True)
    plt.title('Diameters')
    plt.legend(['Height', 'Width', 
        'Least Feret', 'Greatest Feret',
        'Equivalent Circle Perimeter', 'Least Bounding Circle',
        'Horizontal Martin', 'Vertical Martin',
        'Least Bounding Rectangle Width', 'Least Bounding Rectangle Height',
        'Fiber Length', 'Fiber Width',
        'Major Axis', 'Minor Axis'],
        loc='upper left')
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
    mid, low, high = mean_confidence_interval(listECPDiameter, confidence=0.70)
    print(low, mid, high)

    listECADiameter.sort()
    #for idx, x in enumerate(listECADiameter):
    #    listECADiameter[idx] = (x / 2.0)**2 * math.pi
    listLBCDiameter.sort()
    #for idx, x in enumerate(listLBCDiameter):
    #    listLBCDiameter[idx] = (x / 2.0)**2 * math.pi

    #listAreas
    stdev = statistics.pstdev(listAreas)
    variance = statistics.pvariance(listAreas)
    plt.subplot(211), plt.hist(listAreas, bins=10000, density=True, histtype='step', cumulative=True)
    plt.subplot(211), plt.hist(listAreas, bins=10000, density=True, histtype='step', cumulative=-1)
    plt.subplot(211), plt.title('Histogram Area\n' + 'Standard Deviation: ' + str(stdev) + '\n' + 'Variance: ' + str(variance))
    plt.subplot(211), plt.xlabel('μm^2')
    plt.subplot(211), plt.ylabel('Occurence')
    plt.subplot(211), plt.grid(True)

    #listBoundingBoxAreas
    plt.subplot(212), plt.hist(listBoundingBoxAreas, bins=10000, density=True, histtype='step', cumulative=True)
    plt.subplot(212), plt.hist(listBoundingBoxAreas, bins=10000, density=True, histtype='step', cumulative=-1)
    plt.subplot(212), plt.title('Histogram Bounding Box Areas')
    plt.subplot(212), plt.xlabel('μm')
    plt.subplot(212), plt.ylabel('Occurence')
    plt.subplot(212), plt.grid(True)

    #plt.show()

    #listPerimeters
    plt.subplot(311), plt.hist(listECPDiameter, bins=100, density=True, histtype='step', cumulative=True)
    plt.subplot(311), plt.hist(listECPDiameter, bins=100, density=True, histtype='step', cumulative=-1)
    plt.subplot(311), plt.title('Histogram Perimeter')
    plt.subplot(311), plt.xlabel('μm')
    plt.subplot(311), plt.ylabel('Occurence')
    plt.subplot(311), plt.grid(True)

    #listCPM
    plt.subplot(312), plt.hist(listCPM, bins=100, density=True, histtype='step', cumulative=True)
    plt.subplot(312), plt.hist(listCPM, bins=100, density=True, histtype='step', cumulative=-1)
    plt.subplot(312), plt.title('Histogram Convex Perimeter')
    plt.subplot(312), plt.xlabel('μm')
    plt.subplot(312), plt.ylabel('Occurence')
    plt.subplot(312), plt.grid(True)
    
    #listECPDiameter
    plt.subplot(313), plt.hist(listECPDiameter, bins=100, density=True, histtype='step', cumulative=True)
    plt.subplot(313), plt.hist(listECPDiameter, bins=100, density=True, histtype='step', cumulative=-1)
    plt.subplot(313), plt.title('Histogram Equivalent Circular Perimeter Diameter')
    plt.subplot(313), plt.xlabel('μm')
    plt.subplot(313), plt.ylabel('Occurence')
    plt.subplot(313), plt.grid(True)

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

def students_ttest(inputList):
    subplotColumns = 3
    subplotRows = 1
    if (len(inputList) % subplotColumns == 0):
        subplotRows = int(len(inputList) / subplotColumns)
    else:
        subplotRows = int(len(inputList) / subplotColumns) + 1

    ax = plt.subplot(subplotRows, subplotColumns, 1)
    low = 2 * MICRO_METER_PER_PIXEL
    high = 5000#1460
    for idx, rvs in enumerate(inputList):
        rvs = [i for j, i in enumerate(rvs) if j > DIAMETER_THRESHOLD_LOW or j < DIAMETER_THRESHOLD_HIGH]
        listMeasureX0 = list()
        
        for percentile in CDF_PERCENTILE_REF:
            listMeasureX0.append(np.percentile(rvs, percentile))
        tmp = np.array((listMeasureX0, CDF_PERCENTILE_REF))
        tmpRef = np.array((CDF_X0_REF, CDF_PERCENTILE_REF))

        print("Standard Deviation Ref: " + str(tmpRef.std()))
        print("Standard Deviation x0: " + str(tmp.std()))
        plt.subplot(subplotRows, subplotColumns, idx+1, sharex=ax, sharey=ax)
        plt.plot(CDF_X0_REF, CDF_PERCENTILE_REF, 'o', listMeasureX0, CDF_PERCENTILE_REF, '-')
        plt.grid(True)
        plt.xscale('log')
        plt.title(listDebugMeasurementTitles[idx])
        #print(str(listDebugMeasurementTitles[idx]) + ": " + str(scipy.stats.ttest_1samp(rvs, int(X50REF))))
        print(str(listDebugMeasurementTitles[idx]) + ": " + str(scipy.stats.ttest_ind(listMeasureX0, CDF_X0_REF, equal_var=True)))
  
    plt.show()















##### MAIN ####
def main():
    global DEBUG_FILTERING, DEBUG_SEGMENTING, DEBUG_EDITING, DEBUG_PLOT, DEBUG_DATA, DEBUG_STUDENTS_TTEST, FILENAME, X50REF, X503TEMP, CDF_PERCENTILE_REF, EXCEL_ROW
    DEBUG_FILTERING = False
    DEBUG_SEGMENTING = False
    DEBUG_EDITING = False
    DEBUG_PLOT = False
    DEBUG_DATA = False
    DEBUG_STUDENTS_TTEST = True
    DEBUG_TEST_CAL = False
    DEBUG_TEST_ALL = False

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["debug", "plot", "data", "filename=", "x50=", "test-cal", "test-all"])
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
            listTest.append((1040, 394, 441, [2.55,  3.04,  3.50,  3.94,  4.54,  5.27,  5.91,  6.60,
                                               7.30,  7.88,  8.46,  9.20,  10.24, 11.33, 12.45, 14.01,
                                               16.09, 18.92, 22.46, 28.84, 39.86, 55.84, 74.26, 89.83,
                                               98.02, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
            listTest.append((1084, 396, 452, []))
            listTest.append((1108, 449, 486, []))
            listTest.append((1109, 401, 417, []))
            listTest.append((1150, 426, 416, []))
            listTest.append((1156, 422, 435, []))
            listTest.append((1159, 409, 483, []))
            listTest.append((1162, 388, 457, []))
            listTest.append((1168, 426, 542, []))
            listTest.append((1175, 428, 361, []))
            listTest.append((1177, 421, 537, []))
            listTest.append((1181, 439, 525, []))
            listTest.append((1187, 390, 389, []))
            listTest.append((1190, 406, 428, []))
            listTest.append((1191, 409, 517, []))
            listTest.append((1197, 409, 532, []))
            listTest.append((1206, 421, 522, []))
            listTest.append((1207, 356, 459, []))
            listTest.append((1210, 401, 506, []))
            listTest.append((1213, 394, 518, []))
            listTest.append((1214, 431, 437, []))
            listTest.append((1217, 437, 466, []))
            listTest.append((11095, 450, 383, []))
            listTest.append((11096, 414, 477, []))
            listTest.append((11104, 416, 456, []))
            listTest.append((11112, 387, 476, []))
            listTest.append((11133, 391, 472, []))
            listTest.append((11144, 410, 478, []))
            listTest.append((11146, 420, 539, []))
            listTest.append((11147, 405, 579, []))
            listTest.append((11158, 416, 509, []))
            listTest.append((11160, 419, 507, []))
            listTest.append((11199, 434, 465, []))
            listTest.append((11204, 407, 545, []))
            listTest.append((11216, 405, 443, []))
            listTest.append((11222, 415, 453, []))

        EXCEL_ROW = 0
        for (file, temp3, expected, cdfPercentiles) in listTest:
            FILENAME = "images/" + str(file) + "/latest0.jpg"
            X503TEMP = temp3
            X50REF = expected
            if (cdfPercentiles != []):
                CDF_PERCENTILE_REF = cdfPercentiles
            else:
                CDF_PERCENTILE_REF = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            EXCEL_ROW += 1
            analyze()
    else:
        analyze()

    wb.close()

def process(arg):
    print(arg)

if __name__ == "__main__":
    main()
