import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
import transformations

## Helper functions ############################################################

def inbounds(shape, indices):
    '''
        Input:
            shape -- int tuple containing the shape of the array
            indices -- int list containing the indices we are trying 
                       to access within the array
        Output:
            True/False, depending on whether the indices are within the bounds of 
            the array with the given shape
    '''
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    # Implement in child classes
    def detectKeypoints(self, image):
        raise NotImplementedError()



class DummyKeypointDetector(KeypointDetector):
    '''
        Compute silly example features. This doesn't do anything meaningful, but
        may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'. Also compute an 
        # orientation for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")
        Ix = ndimage.sobel(srcImage, 0)  # horizontal derivative
        Iy = ndimage.sobel(srcImage, 1)  # vertical derivative
        IxIx = Ix * Ix
        IyIy = Iy * Iy
        IxIy = Ix * Iy
        wpIxIx = ndimage.gaussian_filter(IxIx, sigma=0.5)
        wpIyIy = ndimage.gaussian_filter(IyIy, sigma=0.5)
        wpIxIy = ndimage.gaussian_filter(IxIy, sigma=0.5)
        harrisImage = wpIxIx * wpIyIy - wpIxIy ** 2 - 0.1 * (wpIxIx + wpIyIy) ** 2
        orientationImage = np.degrees(np.arctan2(Ix, Iy))
        # TODO-BLOCK-END

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")
        height, width = harrisImage.shape[:2]
        newImg = ndimage.maximum_filter(harrisImage, size=7)
        for y in range(height):
            for x in range(width):
                destImage[y, x] = True if harrisImage[y, x] == newImg[y, x] else False
        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.
        # You will need to implement this function.
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                # raise Exception("TODO in features.py not implemented")
                f.pt = (x, y)
                # Dummy size
                f.size = 10
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]
                # TODO-BLOCK-END

                features.append(f)
        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        detector = cv2.ORB_create()
        return detector.detect(image)


# Compute Custom keypoint detector (extra credit)
class CustomKeypointDetector(KeypointDetector):
    # detect n feature points using adaptive non-maximal suppression
    def ANMS(self, features, n=500, cRobust=0.9):
        N = len(features)
        if N <= n:
            return features

        pts = [i.pt for i in features]
        distances = spatial.distance.cdist(pts, pts, 'euclidean')  # distances between each pair of points
        corner_strengths = [i.response for i in features]

        r = [0 for _ in range(N)]
        for i in range(N):
            thresh = corner_strengths[i] / cRobust
            points = [idx for idx in range(N) if corner_strengths[idx] > thresh and idx != i]
            if not points:
                continue
            dists = [distances[i][p] for p in points]
            r[i] = min(dists)
        res_idx = np.argsort(r)[::-1][:n]  # from max to min
        res = [features[i] for i in res_idx]
        return res


    def detectKeypoints(self, image):
        # raise NotImplementedError('NOT IMPLEMENTED')
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        harrisImage, orientationImage = HarrisKeypointDetector.computeHarrisValues(self, grayImage)
        harrisMaxImage = HarrisKeypointDetector.computeLocalMaxima(self, harrisImage)
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue
                f = cv2.KeyPoint()
                f.pt = (x, y)
                f.size = 10
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]
                features.append(f)

        n_val = int(len(features) * 0.05)
        print(len(features), '---', n_val)
        final_features = self.ANMS(features, n=n_val, cRobust=0.9)
        return final_features


## Feature descriptors #########################################################

class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        grayImage_padded = np.pad(grayImage, 2, 'constant', constant_values=0)

        for i, f in enumerate(keypoints):
            x, y = int(f.pt[0]), int(f.pt[1])

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # Note: use grayImage to compute features on, not the input image
            # TODO-BLOCK-BEGIN
            # raise Exception("TODO in features.py not implemented")
            neighborhood = grayImage_padded[y+2-2:y+2+3, x+2-2:x+2+3].reshape(25)
            desc[i] = neighborhood
            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            transMx = np.zeros((2, 3))

            # TODO 5: Compute the transform as described by the feature
            # location/orientation and store in 'transMx.' You will need 
            # to compute the transform from each pixel in the 40x40 rotated 
            # window surrounding the feature to the appropriate pixels in 
            # the 8x8 feature descriptor image. 'transformations.py' has
            # helper functions that might be useful
            # Note: use grayImage to compute features on, not the input image
            # TODO-BLOCK-BEGIN
            # raise Exception("TODO in features.py not implemented")
            x, y = int(f.pt[0]), int(f.pt[1])
            T1 = transformations.get_trans_mx(np.array([-x, -y, 0]))
            R = transformations.get_rot_mx(0, 0, np.radians(-f.angle))
            S = transformations.get_scale_mx(0.2, 0.2, 0)
            T2 = transformations.get_trans_mx(np.array([4, 4, 0]))
            transMx = np.dot(T2, np.dot(S, np.dot(R, T1)))
            transMx = np.delete(transMx, 2, 1)[0:2, 0:3]
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: Normalize the descriptor to have zero mean and unit 
            # variance. If the variance is negligibly small (which we 
            # define as less than 1e-10) then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            #raise Exception("TODO in features.py not implemented")
            destImage = destImage - np.mean(destImage)
            if np.var(destImage) < 10 ** -10:
                destImage = np.zeros(windowSize ** 2)
            else:
                destImage = (destImage / np.std(destImage)).reshape(windowSize ** 2)
            desc[i] = destImage
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        # raise NotImplementedError('NOT IMPLEMENTED')
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        windowSize = 8
        cellSize = 2
        binSize = 90
        numBin = 360 // binSize

        desc = np.zeros((len(keypoints), ((windowSize ** 2) // (cellSize ** 2) * numBin)))

        # quantized orientation
        for i, f in enumerate(keypoints):
            transMx = np.zeros((2, 3))

            x, y = int(f.pt[0]), int(f.pt[1])
            T1 = transformations.get_trans_mx(np.array([-x, -y, 0]))
            R = transformations.get_rot_mx(0, 0, np.radians(-f.angle))
            S = transformations.get_scale_mx(0.2, 0.2, 0)
            T2 = transformations.get_trans_mx(np.array([4, 4, 0]))
            transMx = np.dot(T2, np.dot(S, np.dot(R, T1)))
            transMx = np.delete(transMx, 2, 1)[0:2, 0:3]

            destImage = cv2.warpAffine(grayImage, transMx,
                                       (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            destImage = destImage - np.mean(destImage)
            if np.var(destImage) < 10 ** -10:
                destImage = np.zeros([windowSize, windowSize])
            else:
                destImage = (destImage / np.std(destImage))

            orientationImage = np.zeros((windowSize, windowSize), dtype=float)
            Ix = ndimage.sobel(destImage, 0)
            Iy = ndimage.sobel(destImage, 1)
            orientationImage = np.degrees(np.arctan2(Iy, Ix)) + 179
            gradientMagnitude = np.sqrt(Ix ** 2 + Iy ** 2)

            patchBin = []
            # Iterate cells
            for xCell in range(0, windowSize, cellSize):
                for yCell in range(0, windowSize, cellSize):
                    cellBin = np.zeros(numBin)
                    # Iterate pixels in a cell
                    for xPixel in range(cellSize):
                        for yPixel in range(cellSize):
                            x = xCell + xPixel;
                            y = yCell + yPixel;
                            binIdx = int(orientationImage[x][y] // binSize)
                            cellBin[binIdx] += gradientMagnitude[x][y]
                    patchBin = np.append(patchBin, cellBin)
            desc[i] = patchBin

        return desc


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        raise NotImplementedError

    # Evaluate a match using a ground truth homography. This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        assert desc1.ndim == 2
        assert desc2.ndim == 2
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching. This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")
        n1, n2 = desc1.shape[0], desc2.shape[0]
        distances = spatial.distance.cdist(desc1, desc2, 'euclidean')  # distances between each pair of points
        closest_pts = np.argmin(distances, 1)  # find the closest point in img2 for each point in img1
        for i in range(n1):
            queryIdx = i
            trainIdx = closest_pts[i]
            dist = distances[i][closest_pts[i]]
            f = cv2.DMatch(queryIdx, trainIdx, dist)
            matches.append(f)
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        assert desc1.ndim == 2
        assert desc2.ndim == 2
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image. If the SSD distance is negligibly small, in this case less 
        # than 1e-5, then set the distance to 1. If there are less than two features,
        # set the distance to 0.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")
        n1, n2 = desc1.shape[0], desc2.shape[0]
        distances = spatial.distance.cdist(desc1, desc2, 'euclidean')  # distances between each pair of points
        match_pts = np.argsort(distances, axis=1)
        if n2 < 2:  # less than two features in img2
            for i in range(n1):
                f = cv2.DMatch(i, match_pts[i][0], 0)
                matches.append(f)
        else:
            for i in range(n1):
                first_match = match_pts[i][0]
                second_match = match_pts[i][1]
                first_ssd = distances[i][first_match] if distances[i][first_match] >= 10 ** -5 else 1
                second_ssd = distances[i][second_match] if distances[i][second_match] >= 10 ** -5 else 1
                ratio = first_ssd / second_ssd
                f = cv2.DMatch(i, first_match, ratio)
                matches.append(f)
        # TODO-BLOCK-END

        return matches
