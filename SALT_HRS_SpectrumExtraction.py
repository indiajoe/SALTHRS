""" Pipeline for extracting SALT - HRS instrument spectrum 
    See example.py for how to run this pipeline. 
                                       J.P.Ninan    indiajoe@gmail.com """
import numpy as np
import numpy.ma
from astropy.io import fits
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from skimage.filter import threshold_adaptive
from scipy import ndimage
import scipy.interpolate as interp
import pickle
import os.path

plt.rcParams['keymap.back'] = [u'left', u'backspace']  # To remove back screen togle on c keystrock

class HRSSpectrum:
    def __init__(self,TargetFile,LampTargetFibre,LampSkyFibre,ArcFile):
        """ Input Example
        TargetFile = 'CRmbgphR201412220032.fits'  # CR removed Target source spectrum file
        LampTargetFibre = 'Flats/mbgphR201412180033.fits'  # Flat lamp in Target Fibre
        LampSkyFibre = 'Flats/mbgphR201412220023.fits'  # Flat lamp in Sky Fibre
        """

        self.TargetFile = TargetFile
        self.LampTargetFibre = LampTargetFibre
        self.LampSkyFibre = LampSkyFibre
        self.ArcFile = ArcFile
        self.TargetFlux = dict()  # 1D Array for each apperture in a dictionary
        self.SkyFlux = dict()
        self.ArcFlux = dict()
        self.TargetminusSky = dict()
        self.ShiftScaleSkyLinesAll = dict()
        self.ShiftScaleSkyLineAVG = dict()
        self.PixelVsWavelength = dict()
        self.Wavelength = dict()
        self.ContinuumRegions = dict()
        self.SmoothContinuum = dict()
        self.TargetminusSkyNorm = dict()

    def ThresholdLampImages(self,TargetFibreFile='TargetFibreAppMask.npy',SkyFibreFile='SkyFibreAppMask.npy',bsize=401,offset=0):
        """ If .npy files inpute by TargetFibreFile or SkyFibreFile are not found, then it Generates the threshold mask by adaptive thresholding the lamp image for tracing aperture. This mask will also be written to same filename """
        try :
            self.ThresholdedLampTargetMask = np.load(TargetFibreFile)
            print('Loaded Target Apperture Mask from {0}'.format(TargetFibreFile))
        except(IOError):
            print('Cannot find the file '+TargetFibreFile)
            print('Proceeding to Adaptive thresholding for Target Fiber Aperture')
            self.ThresholdedLampTargetMask = self.ImageThreshold(self.LampTargetFibre,bsize=bsize,offset=offset)
            np.save(TargetFibreFile,self.ThresholdedLampTargetMask)

        try :
            self.ThresholdedLampSkyMask = np.load(SkyFibreFile)
            print('Loaded Sky Apperture Mask from {0}'.format(SkyFibreFile))
        except(IOError):
            print('Cannot find the file '+SkyFibreFile)
            print('Proceeding to Adaptive thresholding for Sky Fiber Aperture')
            self.ThresholdedLampSkyMask = self.ImageThreshold(self.LampSkyFibre,bsize=bsize,offset=offset)
            np.save(SkyFibreFile,self.ThresholdedLampSkyMask)
            

    def ImageThreshold(self,imgfile,bsize=401,offset=0):
        """ Returns adptive threholded image """
        imgArray = fits.getdata(imgfile)
        # Adptive thresholding..
        ThresholdedMask = threshold_adaptive(imgArray, bsize,offset=offset)#, 'mean')
        plt.imshow(np.ma.array(imgArray,mask=~ThresholdedMask))
        plt.colorbar()
        plt.show()
        return ThresholdedMask

        # Commet out the lines below to see simple threshold result
        # LampTargetArray = imgArray
        # LTmedian = np.median(LampTargetArray)
        # LampTargetMasked = np.ma.masked_less(LampTargetArray, LTmedian+3*LTmedian)
        # print(LTmedian)
        # plt.imshow(LampTargetMasked)
        # plt.colorbar()
        # plt.show()


    def LabelAppertures(self, TargetFibreFile='TargetFibreAppLabel.npy',SkyFibreFile='SkyFibreAppLabel.npy',DirectlyEnterRelabel= False):
        """ If .npy files input by TargetFibreFile or SkyFibreFile are not found, then it interactively asks user to label disjoint apperture regions in the mask with sequential labels. This mask will also be written to same filename """
        try :
            self.TargetAppLabel = np.load(TargetFibreFile)
            print('Loaded Target filber apperture labels from {0}'.format(TargetFibreFile))
        except(IOError):
            print('Cannot find the file '+TargetFibreFile)
            print('Proceeding to Interactively label appertures for Target Fiber')
            self.TargetAppLabel = self.LabelDisjointRegions(self.ThresholdedLampTargetMask,DirectlyEnterRelabel= DirectlyEnterRelabel)
            np.save(TargetFibreFile,self.TargetAppLabel)

        try :
            self.SkyAppLabel = np.load(SkyFibreFile)
            print('Loaded Sky filber apperture labels from {0}'.format(SkyFibreFile))
        except(IOError):
            print('Cannot find the file '+SkyFibreFile)
            print('Proceeding to Interactively label appertures for Sky Fiber')
            self.SkyAppLabel = self.LabelDisjointRegions(self.ThresholdedLampSkyMask,DirectlyEnterRelabel= DirectlyEnterRelabel)
            np.save(SkyFibreFile,self.SkyAppLabel)

        
    def LabelDisjointRegions(self,mask,minarea=1000,DirectlyEnterRelabel= False):
        """ Interactively label disjoint regions in a mask """

        labeled_array, num_features = ndimage.label(mask)
#        labeled_array[labeled_array>90] = 0
        NewLabeledArray = labeled_array.copy()
        sugg = 1
        print('Regions of area less than {0} are discarded'.format(minarea))
        for label in range(1,np.max(labeled_array)+1):
            size = np.sum(labeled_array==label)
            if size < minarea : # Discard label if the region size is less than min area
                NewLabeledArray[labeled_array==label] = 0
                    

        if not DirectlyEnterRelabel:
            print('Enter q to discard all remaining regions')
            for label in np.unique(NewLabeledArray):
                if label == 0: # Skip 0 label
                    continue
                plt.clf()
                # plt.imshow(np.ma.array(LampTargetArray,mask=~(labeled_array==label)))
                # plt.colorbar()
                plt.imshow(np.ma.array(labeled_array,mask=~(labeled_array==label)))
                plt.show(block=False)
                print('Area of region = {0}'.format(size))
                print('Current Label : {0}'.format(label))
                newlbl = sugg
                uinput = raw_input('Enter New Region label  (default: {0}): '.format(sugg)).strip()
                if uinput: #User provided input
                    if uinput == 'q':
                        # Set all the labels current and above this value to Zero and exit
                        NewLabeledArray[labeled_array >= label]= 0
                        break
                    else:
                        newlbl = int(uinput)

                NewLabeledArray[labeled_array==label] = newlbl
                sugg = newlbl + 1

        else:
            plt.imshow(NewLabeledArray)
            plt.colorbar()
            plt.show(block=False)
            print('Create a file with the label renames to be executed.')
            print('File format should be 2 columns :  OldLabel NewLabel')
            print('Note: If you want to keep old name but also merge another region to same label, give that first')
            uinputfile = raw_input('Enter the filename :').strip()
            if uinputfile: #User provided input
                with open(uinputfile,'r') as labelchangefile:
                    labelchangelist = [tuple(int(i) for i in entry.rstrip().split()) for entry in labelchangefile \
                                       if len(entry.rstrip().split()) == 2]
                print('Label Renaming :',labelchangelist)
                for oldlabel, newlabel in labelchangelist:
                    # Remove existance of any pixels by new label
                    NewLabeledArray[labeled_array == newlabel] = 0  #IMP: this makes the order in which given to merge new region important.. Think...
                    # Now assign new label
                    NewLabeledArray[labeled_array == oldlabel] = newlabel
                    
                # Remove all other labels..
                NewLabelList = [newlabel for oldlabel, newlabel in labelchangelist]
                for label in np.unique(NewLabeledArray):
                    if label not in NewLabelList:
                        NewLabeledArray[labeled_array==label] = 0

            else:
                print('No input given. No relabelling done..')

            
            
        plt.clf()
        print('New labelled regions')
        plt.imshow(NewLabeledArray)
        plt.colorbar()
        plt.show()

        return NewLabeledArray

    def ExtractTargetSpectrum(self,apertures=None,ShowPlot=True):
        """ Extracts the spectrum by summing along column for the input list of appertures """
        print('Extracting Target Spectrum')
        if apertures is None : apertures = np.arange(1,np.max(self.TargetAppLabel)+1)
        TargetImageArray = fits.getdata(self.TargetFile)
        for aper in apertures:
            print('Aperture : {0}'.format(aper))
            self.TargetFlux[aper] = np.ma.sum(np.ma.array(TargetImageArray,mask=~(self.TargetAppLabel==aper)),axis=0)
            if ShowPlot:
                plt.plot(self.TargetFlux[aper])
                plt.show()

    def ExtractSkySpectrum(self,apertures=None,ShowPlot=True):
        """ Extracts the spectrum by summing along column for the input list of appertures """
        print('Extracting Sky Spectrum')
        if apertures is None : apertures = np.arange(1,np.max(self.SkyAppLabel)+1)
        TargetImageArray = fits.getdata(self.TargetFile)
        for aper in apertures:
            print('Aperture : {0}'.format(aper))
            self.SkyFlux[aper] = np.ma.sum(np.ma.array(TargetImageArray,mask=~(self.SkyAppLabel==aper)),axis=0)
            if ShowPlot:
                plt.plot(self.SkyFlux[aper])
                plt.show()

    def ExtractArcSpectrum(self,apertures=None,ShowPlot=True, useskyapp = False):
        """ Extracts the spectrum by summing along column for the input list of appertures 
        useskyapp = True will extract the arc spectrum from sky appertures.
        If arc is available through target spectrum, keep default useskyapp = False"""
        print('Extracting Arc Spectrum')
        if apertures is None : apertures = np.arange(1,np.max(self.SkyAppLabel)+1)
        ArcImageArray = fits.getdata(self.ArcFile)

        if useskyapp:
            AppToUse = self.SkyAppLabel
        else:
            AppToUse = self.TargetAppLabel

        for aper in apertures:
            print('Aperture : {0}'.format(aper))
            self.ArcFlux[aper] = np.ma.sum(np.ma.array(ArcImageArray,mask=~(AppToUse==aper)),axis=0)
            if ShowPlot:
                plt.plot(self.ArcFlux[aper])
                plt.show()
        
    def AlignScaleSkyLines(self,apertures=None,PickledFile='LatestAlignScaleAVG.pickle'):
        """ Align and Scale the Sky emission lines Interactively to Target Star spectrum lines 
        for the input list of appertures """
        if os.path.isfile(PickledFile):
            with open(PickledFile, 'rb') as fls:
                self.ShiftScaleSkyLineAVG = pickle.load(fls)
            print('Loaded previously fitted shifts and scaling from {0}'.format(PickledFile))
            return

        print('Aligning and Scaling Sky spectrum Interactively to Target Spectrum')
        if apertures is None : apertures = np.arange(1,min(np.max(self.TargetAppLabel),np.max(self.SkyAppLabel))+1)
        for aper in apertures:
            print('Aperture : {0}'.format(aper))
            ListofPositions=[]
            ListofShifts=[]
            ListofScaleFactors=[]
            ListOfSkyLines = self.SelectLinesInteractively(np.arange(len(self.SkyFlux[aper])),self.SkyFlux[aper])
            for line in ListOfSkyLines:
                SkyPos = line.mean_0.value
                SkyAmp = line.amplitude_0.value
                SkyArea = line.amplitude_0.value * abs(line.stddev_0.value) *np.sqrt(np.pi)
                
                # Now fit The Target source line at this position
                Tlinefit = FitLineToData(np.arange(len(self.TargetFlux[aper])),self.TargetFlux[aper],
                                         SkyPos,SkyAmp,AmpisBkgSubtracted=True)
                TarPos = Tlinefit.mean_0.value
                TarAmp = Tlinefit.amplitude_0.value
                TarArea = Tlinefit.amplitude_0.value * abs(Tlinefit.stddev_0.value) *np.sqrt(np.pi)
                
                Shift = TarPos - SkyPos
                ScaleFactor = TarArea / SkyArea
                print('Line :{0} | shift: {1} | ScaleFactor :{2}'.format(TarPos,Shift,ScaleFactor))
                ListofPositions.append(TarPos)
                ListofShifts.append(Shift)
                ListofScaleFactors.append(ScaleFactor)

            self.ShiftScaleSkyLinesAll[aper] = [ListofPositions,ListofShifts,ListofScaleFactors]
            self.ShiftScaleSkyLineAVG[aper] =(np.mean(ListofShifts),np.mean(ListofScaleFactors))
            print('Aper :{0} > Mean Shift ={1} ; Mean Scale ={2}'.format(aper,self.ShiftScaleSkyLineAVG[aper][0],self.ShiftScaleSkyLineAVG[aper][1]))

        # Saving the latest fit in a pickled file.
        with open(PickledFile, 'wb') as fls:
            pickle.dump(self.ShiftScaleSkyLineAVG, fls)


    def IdentifyArcLines(self,apertures=None,PickledFile='LatestArcLineIdentify.pickle'):
        """ Interactively Identify the lines in Arc lamp spectrum to wavelenth to obtain wavelenth calibration 
        for the input list of appertures """
        if os.path.isfile(PickledFile):
            with open(PickledFile, 'rb') as fls:
                self.PixelVsWavelength = pickle.load(fls)
            print('Loaded previously identified lines from {0}'.format(PickledFile))
            return

        print('Interactively Identifying wavelengths of lines in Arc Spectrum')
        if apertures is None : apertures = np.arange(1,min(np.max(self.TargetAppLabel),np.max(self.SkyAppLabel))+1)
        for aper in apertures:
            print('Aperture : {0}'.format(aper))
            ListofPositions=[]
            ListofWavel=[]
            ListOfArcLines = self.SelectLinesInteractively(np.arange(len(self.ArcFlux[aper])),self.ArcFlux[aper],ExtraUserInput='Wavelength')
            for line,wavel in ListOfArcLines:
                ListofPositions.append(line.mean_0.value)
                ListofWavel.append(float(wavel))

            # Sort in the increasing order of wavelength
            IndexToSort = np.argsort(ListofWavel)
            ListofPositions = np.array(ListofPositions)[IndexToSort]
            ListofWavel = np.array(ListofWavel)[IndexToSort]

            self.PixelVsWavelength[aper] = (ListofPositions,ListofWavel)

        # Saving the latest Identification in a pickled file.
        with open(PickledFile, 'wb') as fls:
            pickle.dump(self.PixelVsWavelength, fls)
        
        
    def SkySubtract(self,apertures=None,ShowPlot=True):
        """ Subtracts sky from Target spectrum for the input list of appertures """
        print('Target Spectrum minus Sky spectrum')
        if apertures is None : apertures = np.arange(1,min(np.max(self.TargetAppLabel),np.max(self.SkyAppLabel))+1)
        for aper in apertures:
            print('Aperture : {0}'.format(aper))

            tck=interp.splrep(np.arange(len(self.SkyFlux[aper])),self.SkyFlux[aper],s=0)
            shift = self.ShiftScaleSkyLineAVG[aper][0]
            scale = self.ShiftScaleSkyLineAVG[aper][1]
            AlignedScaledSky = interp.splev(np.arange(len(self.SkyFlux[aper]))-shift,tck,der=0) *scale

            self.TargetminusSky[aper] = self.TargetFlux[aper] - AlignedScaledSky
            if ShowPlot:
                plt.plot(self.TargetminusSky[aper])
                plt.show()

    def WavelengthCalibrateTarget(self,apertures=None,method='p2',ShowPlot=True, applyskyshift= False):
        """ Calculates the wavelength calibration of the spectra in each aperture 
        method : spline  # To fit spline curve to all points
               : p2     # To fit 2nd order polynomial   (pi where i is an integer will fit i^th order poly)
        applyskyshift = False, doesnot shift the wavelength calibration. Set it to true if arc lamp was taken in sky fibre."""
        print('Doing Wavelength calibration')
        if apertures is None : apertures = np.arange(1,min(np.max(self.TargetAppLabel),np.max(self.SkyAppLabel))+1)
        for aper in apertures:
            print('Aperture : {0}'.format(aper))

            if applyskyshift :
                shift = self.ShiftScaleSkyLineAVG[aper][0]
            else:
                shift = 0

            if method == 'spline' : # Fit spline
                tck=interp.splrep(self.PixelVsWavelength[aper][0],self.PixelVsWavelength[aper][1],s=0)
                self.Wavelength[aper] = interp.splev(np.arange(len(self.TargetminusSky[aper]))-shift,tck,der=0)
            elif method[0] == 'p' :  # Fit polynomial of asked order
                order = int(method[1:])
                p = np.polyfit(self.PixelVsWavelength[aper][0],self.PixelVsWavelength[aper][1],order)
                self.Wavelength[aper] = np.polyval(p,np.arange(len(self.TargetminusSky[aper]))-shift)

            if ShowPlot:
                plt.plot(np.arange(len(self.TargetminusSky[aper]))-shift,self.Wavelength[aper])
                plt.plot(self.PixelVsWavelength[aper][0],self.PixelVsWavelength[aper][1],'o')
                plt.title('Wavelength fitting')
                plt.show()
                plt.title('Wavelength Calibrated Spectrum')
                plt.plot(self.Wavelength[aper],self.TargetminusSky[aper])
                plt.show()
            
    def SelectLinesInteractively(self,SpectrumX,SpectrumY,ExtraUserInput=None, LineSigma=3):
        """ Fits Gaussian to all points user press m and c to confirm
        ExtaUserInput : (str) If provided will ask user from the extra input for each line 
                            Example: 'Wavelength'

        LineSigma : Approximated Sigma width of the line in pixels. 
                    Size of the window region to use for fitting line will be 5*LineSigma.
        Returns : List of Fitted Line Models (and user input if any).
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Press m to select line, then c to confirm the selection.')
        ax.plot(SpectrumX,SpectrumY,linestyle='--', drawstyle='steps-mid',marker='.',color='k')
        junk, = ax.plot([],[])
        PlotedLines=[junk]
        LinesConfirmedToReturn = []
        LatestLineModel = [None]
        # Define the function to run while key is pressed
        def on_key(event):
            if event.key == 'm' :
                Xpos = event.xdata
                Model_fit = FitLineToData(SpectrumX,SpectrumY,Xpos,event.ydata,AmpisBkgSubtracted=False,Sigma = LineSigma)
                if PlotedLines:
                    ax.lines.remove(PlotedLines[-1])  # Remove the last entry in the plotedlines
                IndxCenter = NearestIndex(SpectrumX,Model_fit.mean_0.value)
                SliceToPlotX = SpectrumX[IndxCenter-4*LineSigma:IndxCenter+4*LineSigma+1]
                linefit, =  ax.plot(SliceToPlotX,Model_fit(SliceToPlotX),color='r')
                PlotedLines.append(linefit)
                ax.figure.canvas.draw()

                LatestLineModel[0] = Model_fit
                print(Model_fit)


            elif event.key == 'c' :
                print('Fit acceptence confirmed.')
                junk, = ax.plot([],[])
                PlotedLines.append(junk)  # Add a new junk plot to end of ploted lines

                if ExtraUserInput is not None:  # Ask user for the extra input
                    Uinput = raw_input(ExtraUserInput+ ' : ')
                    LinesConfirmedToReturn.append((LatestLineModel[0],Uinput))
                else:
                    LinesConfirmedToReturn.append(LatestLineModel[0])
                
        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        
        return LinesConfirmedToReturn
        

    def NormalizeContinuum(self,apertures=None,ShowPlot=True,PickledFile='LatestContinuumRegions_Smooth_NormSpectra.pickle'):
        """ Interactively mask the regions which contain lines for fitting a smooth continuum and finally divide by 
        the smooth continuum to normalise the spectrum for the input list of appertures """
        if os.path.isfile(PickledFile):
            with open(PickledFile, 'rb') as fls:
                self.ContinuumRegions,self.SmoothContinuum,self.TargetminusSkyNorm = pickle.load(fls)
            print('Loaded previously marked regions of smooth continuum, Fit as well as normalised spectrum from {0}'.format(PickledFile))
            return

        print('Interactively mask regions of Target spectrum containing lines')
        if apertures is None : apertures = np.arange(1,np.max(self.TargetAppLabel)+1)
        for aper in apertures:
            print('Aperture : {0}'.format(aper))
            self.SmoothContinuum[aper], self.ContinuumRegions[aper] = self.FindSmoothContinuum(self.Wavelength[aper],self.TargetminusSky[aper])

            self.TargetminusSkyNorm[aper] = self.TargetminusSky[aper]/self.SmoothContinuum[aper]

            if ShowPlot:
                plt.plot(self.Wavelength[aper],self.TargetminusSkyNorm[aper])
                plt.show()

        # Saving the latest fitting in a pickled file.
        with open(PickledFile, 'wb') as fls:
            pickle.dump((self.ContinuumRegions,self.SmoothContinuum,self.TargetminusSkyNorm), fls)
            
    def FindSmoothContinuum(self,SpectrumX,SpectrumY):
        """ Interactively finds smooth continuum for the input spectrum 
        Retuns the intepolated smooth continumm as well as the regions used for continuum"""
        # First do a simple thresholding to help user. he can accpet or reject it.
        # Adptive thresholding..
        RunningMedian = ndimage.filters.median_filter(SpectrumY,size=400)
        Residue = SpectrumY - RunningMedian
        MADofResidue = np.median( np.abs(Residue - np.median(Residue)))
        ContinuumRegion = np.abs(Residue) > (MADofResidue *1.4826 * 10)
        UserNotSatisfied = True
        while UserNotSatisfied :
            # Get mask interactively
            ContinuumRegion = self.SelectRegionsInteractively(SpectrumX,SpectrumY,InitialMask=ContinuumRegion)
            # Now fit a smooth continuum
            l = len(SpectrumX[~ContinuumRegion])
#            sm=(l+np.sqrt(2*l))*6   #Smoothing factor for fitting.
            sm = (MADofResidue *1.4826)**2 * l
            print('Crude estimate of smoothing factor: {0}'.format(sm))
            sm=float(raw_input('Enter smoothing factor to apply for spline interpolation : ').strip())
            tck=interp.splrep(SpectrumX[~ContinuumRegion],SpectrumY[~ContinuumRegion],s=sm)
            Scontinuum = interp.splev(SpectrumX,tck)
            plt.plot(SpectrumX,SpectrumY,linestyle='--', alpha=0.5,drawstyle='steps-mid',marker='.',color='k')
            plt.plot(SpectrumX,Scontinuum,color='r')
            plt.show()
            userinp = raw_input('Enter wq to accept the smooth continuum (anything else to repeat) :')
            if userinp == 'wq' :
                UserNotSatisfied = False
        
        return Scontinuum,ContinuumRegion
            
        
            
    def SelectRegionsInteractively(self,SpectrumX,SpectrumY,InitialMask=None):
        """ Interactively asks user to Press q to mark left side and w to mark right side. m to mask, or c to clear.
        In order to generate the mask which marks regions
        Returns : Final Boolean Mask which marks the regions with True.
        """
        if InitialMask is None :
            LineMask = np.zeros(len(SpectrumX))
        else:
            LineMask = InitialMask

        #plot the original spectrum
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Press q to mark left side and w to mark right side. m to mask, or c to clear.')
        ax.plot(SpectrumX,SpectrumY,linestyle='--', alpha=0.5,drawstyle='steps-mid',marker='.',color='k')
        MaskedLineplot,=ax.plot(SpectrumX,np.ma.array(SpectrumY,mask=LineMask),color='r',marker='.')
        Latestmaskplot = [MaskedLineplot]
        NewRegion=[0,0]  # X and Y coordinated of the new region user wants to mask or unmask
        # Define the function to run while key is pressed
        def on_key(event):
            if event.key == 'q' :
                NewRegion[0] = NearestIndex(SpectrumX,event.xdata)
                print('Set left side of region at {0}:{1}'.format(NewRegion[0],event.xdata))
            elif event.key == 'w' :
                NewRegion[1] = NearestIndex(SpectrumX,event.xdata)
                print('Set right side of region at {0}:{1}'.format(NewRegion[1],event.xdata))
            elif event.key == 'm' :
                #Sanity check
                if NewRegion[0] < NewRegion[1]:
                    print('Masking the region of spectrum in {0}:{1} ({2}:{3})'.format(NewRegion[0],NewRegion[1],SpectrumX[NewRegion[0]],SpectrumX[NewRegion[1]]))
                    LineMask[NewRegion[0]:NewRegion[1]+1] = 1
                else:
                    print('{0} should be less than {1}'.format(NewRegion[0], NewRegion[1]))
            elif event.key == 'c' :
                #Sanity check
                if NewRegion[0] < NewRegion[1]:
                    print('Removing mask in the region of spectrum in {0}:{1} ({2}:{3})'.format(NewRegion[0],NewRegion[1],SpectrumX[NewRegion[0]],SpectrumX[NewRegion[1]]))
                    LineMask[NewRegion[0]:NewRegion[1]+1] = 0
                else:
                    print('{0} should be less than {1}'.format(NewRegion[0], NewRegion[1]))

            # Remove the previous plot and update the plot wit new one.
            ax.lines.remove(Latestmaskplot[0])  # Remove the last entry in the plotedlines
            
            MaskedLineplot,=ax.plot(SpectrumX,np.ma.array(SpectrumY,mask=LineMask),color='r',marker='.')
            Latestmaskplot[0] = MaskedLineplot
            ax.figure.canvas.draw()
                            
        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        
        return LineMask

                    



#--------------------#Some extra functions#-----------------#
def NearestIndex(Array,value):
    """ Returns the index of element in numpy 1d Array nearest to value """
    return np.abs(Array-value).argmin()

def FitLineToData(SpecX,SpecY,Pos,Amp,AmpisBkgSubtracted=True,Sigma = 3):
    """ Fits a model line to the SpecX SpecY data 
    Model as well as parameters to fit are defined inside this function
    Sigma = 3 # Approximate size of the line width sigma in pixels.
    Half of the window size to use for fitting line in pixels will be FitWindowH = 5*Sigma  """

    FitWindowH = 5*Sigma  # window to fit the line is 5* expected Sigma of line
    StartIdx = NearestIndex(SpecX,Pos) - FitWindowH
    EndIdx = NearestIndex(SpecX,Pos) + FitWindowH
    SliceToFitX = SpecX[StartIdx:EndIdx+1] 
    SliceToFitY = SpecY[StartIdx:EndIdx+1]
    MedianBkg = np.median(SliceToFitY)
    dw = np.abs(np.median(SliceToFitX[1:]-SliceToFitX[:-1]))
    if not AmpisBkgSubtracted:
        Amp = Amp - MedianBkg

    #Define the line model to fit 
    LineModel = models.Gaussian1D(amplitude=Amp, mean=Pos, stddev=Sigma*dw)+models.Linear1D(slope=0,intercept=MedianBkg)

    #Define fitting object
    Fitter = fitting.LevMarLSQFitter()#SLSQPLSQFitter()
    
    #Fit the model to data
    Model_fit = Fitter(LineModel, SliceToFitX, SliceToFitY)  
    return Model_fit
    

        
###-------------------------------------------------------###
