import queue

class Rect:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top

    def area(self):
        return self.width() * self.height()

def getOverlapRect(rect1, rect2):
    return Rect(
        max(rect1.left, rect2.left),        # left
        max(rect1.top, rect2.top),          # top
        min(rect1.right, rect2.right),      # right
        min(rect1.bottom, rect2.bottom)     # bottom
    )

def mergeRects(rects, overlap_rate=0.9, min_overlap_cnt=8):
    Q = queue.Queue()
    last_update = 0
    last_access = 0

    for x, y, w, h in rects:
        Q.put([Rect(x, y, x+w, y+h), 1])
        last_update += 1

    while last_access < last_update:
        r1, oc1 = Q.get()
        last_access += 1
        updated = False

        last_access_2 = last_access
        while last_access_2 < last_update:
            r2, oc2 = Q.get(); last_access_2 += 1
            a1 = r1.area()
            a2 = r2.area()
            # get overlap rect
            ro = getOverlapRect(r1, r2)
            # get overlap area
            ao = ro.area()
            if ao >= min(a1, a2) * overlap_rate:
                mr = Rect(
                    (r1.left + r2.left) / 2, # left
                    (r1.top + r2.top) / 2, # top
                    (r1.right + r2.right) / 2, # right
                    (r1.bottom + r2.bottom) / 2 # bottom
                )

                last_access += 1 # r2 removed
                Q.put([mr, oc1+oc2]); last_update += 1
                updated = True
                break
            Q.put([r2, oc2])

        if not updated:
            if oc1 >= min_overlap_cnt:
                Q.put([r1, oc1])
    
    mergedRects = []
    while not Q.empty():
        rect, _ = Q.get()
        mergedRects.append([
            int(rect.left),
            int(rect.top),
            int(rect.width()),
            int(rect.height())
        ])
    return mergedRects