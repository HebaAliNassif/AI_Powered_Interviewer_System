import numpy as np

def enum(**enums):
    return type('Enum', (), enums)

HaarLikeFeatureType = enum(
                    TWO_VERTICAL=(1, 2),
                    TWO_HORIZONTAL=(2, 1),
                    THREE_HORIZONTAL=(3, 1),
                    THREE_VERTICAL=(1, 3),
                    FOUR=(2, 2))
HaarLikeFeatureTypes = [HaarLikeFeatureType.TWO_VERTICAL, HaarLikeFeatureType.TWO_HORIZONTAL, HaarLikeFeatureType.THREE_VERTICAL, HaarLikeFeatureType.THREE_HORIZONTAL, HaarLikeFeatureType.FOUR]


class RectangleRegion:
    def __init__(self, top_left, bottom_right, white):
        assert type(top_left) == tuple
        assert type(bottom_right) == tuple
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.white = white
        
    def compute_feature(self, ii):
        """
        Computes the value of the Rectangle Region given the integral image
        Args:
            integral image : numpy array, shape (m, n)
            x: x coordinate of the upper left corner of the rectangle
            y: y coordinate of the upper left corner of the rectangle
            width: width of the rectangle
            height: height of the rectangle
        """
        x_a = self.top_left[1]-1
        y_a = self.top_left[0]-1
        
        x_b = self.bottom_right[1]
        y_b = self.bottom_right[0]
        
        if (x_a) < 0 and y_a < 0:
            sum_val = ii[y_b, x_b]
        elif (x_a) < 0:
            sum_val = ii[y_b, x_b] - ii[y_a, x_b]
        elif (y_a) < 0:
            sum_val = ii[y_b, x_b] - ii[y_b, x_a]
        else:
            sum_val = ii[y_b, x_b] + ii[y_a, x_a] - ii[y_a, x_b] - ii[y_b, x_a]
        
        if self.white:
            return sum_val
        else:
            return sum_val * -1
    
    def __str__(self):
        return "(top_left.x= %d, top_left.y= %d, bottom_right.x= %d, bottom_right.y= %d)" % (self.top_left[1], self.top_left[0], self.bottom_right[1], self.bottom_right[0])
    def __repr__(self):
        return "[(%d, %d), (%d, %d)]" % (self.top_left[0], self.top_left[1], self.bottom_right[0], self.bottom_right[1])
    
def haar_like_feature_coord(height, width, verbose=False):
        if verbose:
            print('Creating haar-like features..')
        features = []
        types = []
        for feature_typ in HaarLikeFeatureTypes:
            for y in range(height):
                for x in range(width):
                    for dy in range(1, height+1):
                        for dx in range(1, width+1):
                            rects = list()
                            if feature_typ == HaarLikeFeatureType.TWO_VERTICAL and (y + dy <= height and x + 2 * dx <= width):
                                rects.append(RectangleRegion((y, x), (y + dy - 1, x + dx - 1), 1))
                                rects.append(RectangleRegion((y, x + dx), (y + dy - 1, x + 2 * dx - 1), 0))
                                features.append(list(rects))
                                types.append(HaarLikeFeatureType.TWO_VERTICAL)
                            elif feature_typ == HaarLikeFeatureType.TWO_HORIZONTAL  and (y + 2 * dy <= height and x + dx <= width):
                                rects.append(RectangleRegion((y, x), (y + dy - 1, x + dx - 1), 1))
                                rects.append(RectangleRegion((y + dy, x), (y + 2 * dy - 1, x + dx - 1), 0))
                                features.append((rects))
                                types.append(HaarLikeFeatureType.TWO_HORIZONTAL)
                            elif feature_typ == HaarLikeFeatureType.THREE_HORIZONTAL  and (y + 3 * dy <= height and x + dx <= width):
                                rects.append(RectangleRegion((y, x), (y + dy - 1, x + dx - 1), 1))
                                rects.append(RectangleRegion((y + dy, x), (y + 2 * dy - 1, x + dx - 1), 0))
                                rects.append(RectangleRegion((y + 2 * dy, x), (y + 3 * dy - 1, x + dx - 1), 1))
                                features.append((rects))
                                types.append(HaarLikeFeatureType.THREE_HORIZONTAL)
                            elif feature_typ == HaarLikeFeatureType.THREE_VERTICAL  and (y + dy <= height and x + 3 * dx <= width):
                                rects.append(RectangleRegion((y, x), (y + dy - 1, x + dx - 1), 1))
                                rects.append(RectangleRegion((y, x + dx), (y + dy - 1, x + 2 * dx - 1), 0))
                                rects.append(RectangleRegion((y, x + 2 * dx), (y + dy - 1, x + 3 * dx - 1), 1))
                                features.append((rects))
                                types.append(HaarLikeFeatureType.THREE_VERTICAL)
                            elif feature_typ == HaarLikeFeatureType.FOUR  and (y + 2 * dy <= height and x + 2 * dx <= width):
                                rects.append(RectangleRegion((y, x), (y + dy - 1, x + dx - 1), 1))
                                rects.append(RectangleRegion((y, x + dx), (y + dy - 1, x + 2 * dx - 1), 0))
                                rects.append(RectangleRegion((y + dy, x), (y + 2 * dy - 1, x + dx - 1), 0))
                                rects.append(RectangleRegion((y + dy, x + dx), (y + 2 * dy - 1, x + 2 * dx - 1), 1))
                                features.append(list(rects))
                                types.append(HaarLikeFeatureType.FOUR)
        if verbose:
            print('..done. ' + str(len(features)) + ' features created.\n')
        
        return np.asarray(features, dtype=object), np.asarray(types)

def haar_like_feature(ii, height, width, features, verbose=False):
    features_values = []
    for feature in features:
        n_rectangles = len(feature)
        val = 0
        for rect in feature:
            val += rect.compute_feature(ii)
        features_values.append(val)
    return features_values