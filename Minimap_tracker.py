import cv2, pafy
import numpy as np
from PIL import ImageGrab
import json

## to simplify color conversion
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## load video from youtube
def load_yt(url):
    print(f'start loading from {url}')
    vpafy = pafy.new(url)
    best = vpafy.getbest("any", False)
    print('title:',best.title)
    print('filesize',best.get_filesize())
    print('quality',best.resolution)
    return best.url

class Tracker:
    def __init__(self, name):
        self.name = name
        self._filename = f'champion_icons/{self.name}.jpg'
        self._icon = cv2.imread(self._filename)
        self._icon = cv2.resize(self._icon, (24, 24), interpolation = cv2.INTER_AREA)
        self._icon_gray = cv2.cvtColor(self._icon, cv2.COLOR_BGR2GRAY)
        self.path = []
        self.counter = 0
        
# If a champion is detected, then return it's position, else, return None
    def track(self, image):
        assert image.shape == (240, 240),'image shape should be 240x240'
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.position = None
        res = cv2.matchTemplate(image, self._icon_gray, cv2.TM_CCOEFF_NORMED)
        t_maxind = np.unravel_index(res.argmax(), res.shape)
        t_maxpos = res[t_maxind]
        if (t_maxpos >= 0.7):
            self.position = (int(t_maxind[1]) + 12, int(t_maxind[0]) + 12)
        if self.position:
            self.path.append(self.position)
        return self.position

# track multiple champions at once.
class MultiTracker:
    def __init__(self, targets):
        self._trackers = [ Tracker(i) for i in targets]
        
    def track(self, image):
        result = dict()
        for i in self._trackers:
            pos = i.track(image)
            result[i.name] = pos
        return result

# detect the minimap location, if the video is not given, then detect user's screen instead.
def detect_minimap(video = None):
# load minmap image
    minimap = cv2.imread('minimap1.png')    
# convert into grayscale
    minimap = grayscale(minimap)
# canny edge detection
    minimap_edges_og = cv2.Canny(minimap, 25, 50)
    map_found = False
    map_position = (0,0,0,0)

# keep finding till the map is found.
    while(not map_found):
        if not video:
            frame = np.array(ImageGrab.grab())
        else:
            _, frame = video.read()
        
# minimap is at the bottom of the screen, so we only detect the bottom part.
        mid_line = int(0.65 * frame.shape[0])
        bottom = frame[int(mid_line):,:,:]

        bottom = grayscale(bottom)
        edges = cv2.Canny(bottom, 25, 50)
        minimap_edges = minimap_edges_og.copy()
        temp = None
        maxres = None
        probs = dict()
        if(minimap_edges.shape[0] > edges.shape[0]):
            minimap_edges = cv2.resize(minimap_edges_og, (edges.shape[0], edges.shape[0]))

# change the minimap size untill we find its location.
        for i in range(minimap_edges.shape[0], 120, -2):
            res = cv2.matchTemplate(edges , minimap_edges, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.2)
            for pt in zip(*loc[::-1]):
                temp = (pt[0], pt[1]+mid_line, minimap_edges.shape[0]+pt[0], mid_line + minimap_edges.shape[1]+pt[1])
                maxres = res.max()
                cv2.rectangle(frame, temp[:2],temp[2:4],(255,128,255),2)
                probs[maxres] = temp
            minimap_edges = cv2.resize(minimap_edges_og, (i, i))
        if len(probs) > 0:
            map_position = probs[maxres]
            map_found = True
        print('minimap found')
        return map_position


def track_champs(champ_list, mappos, video = None, save_name = None, max_frame = None):
    ''' 
    track champions in the champ_list.
    champ_list: list of champion names, e.g. ['Ahri','Gragas'...]
    mappos: list or tuple. position of minimap.  [top_left.x, top_left.y, bottom_right.x, bottom_right.y]
    video: VideoCapture object. if None is give, then capture user's screen instead
    save_name: string. if given, then save the champion position json file and tracked video to current directory.
    max_frame: int. if given, then stop tracking after max_frame. if not, then continue untill video ends or user presses 'q'. 
    '''
    first_frame = True
    trackers = MultiTracker(champ_list)
    counter = 0
    path = dict()
    if save_name:
        vidfcc = cv2.VideoWriter_fourcc(*'XVID')
        vidwriter  = cv2.VideoWriter(save_name+'.mp4',vidfcc, 30.0, (240, 240))
    while(True):
        if not video:
            frame = np.array(ImageGrab.grab())
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            _, frame = video.read()
        if np.all(frame) == None:
            break
        
        maparea = frame[mappos[1]:mappos[3],mappos[0]:mappos[2]]
        maparea = cv2.resize(maparea, (240, 240))
        showcase = maparea.copy()
        maparea = grayscale(maparea)

        if first_frame:
            prev_frame = maparea
            first_frame = False

        diff = cv2.absdiff(prev_frame, maparea)
        kernel = np.ones((5,5), np.uint8)
        diff_close = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
        masked_close = cv2.bitwise_and(maparea,maparea,mask=diff_close)

        circle_map = np.zeros_like(maparea)
        circles = cv2.HoughCircles(masked_close, cv2.HOUGH_GRADIENT, 1, minDist = 15,param1 = 20, param2 = 10, minRadius = 12, maxRadius= 14)
        if np.all(circles):
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(circle_map, tuple(i[:2]), i[2], (255,255,255), -1)
        circle_mask = np.uint8(circle_map / 255) * 255
        masked_map = cv2.bitwise_and(maparea, maparea, mask = circle_mask)
        poses = trackers.track(masked_map)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for k, v in poses.items():
            if v:
                cv2.putText(showcase, k, (v[0]-12,v[1]+24) , font, 0.3, (255,255,255), 1, cv2.LINE_AA)
                cv2.circle(showcase, v, 12, (12, 255,255), 1)
                val = tuple([counter]+list(v)) # (frame, x_pos, y_pos)
                if k not in path.keys():
                    path[k] = [val]
                else:
                    path[k].append(val)

        if save_name:
            vidwriter.write(showcase)
        if len(showcase.shape) == 2:
            showcase = cv2.cvtColor(showcase, cv2.COLOR_GRAY2BGR)
        cv2.imshow('show',showcase)
        prev_frame = maparea
        
        if max_frame:
            if counter >= max_frame:
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1
    if save_name:
        with open(save_name+'.json','w') as f:
            dump = json.dumps(path)
            f.write(dump)
        vidwriter.release()
    return path

################################################
########### Change the champ_list and video url ############
champ_list = ['Pantheon','Lee Sin','Cassiopeia','Kalista','Gragas','Poppy','Jarvan IV','Ekko','Lucian','Nami']
vid = cv2.VideoCapture(load_yt('https://www.youtube.com/watch?v=rV1yYGnIFsA'))
mappos = detect_minimap(vid)
path = track_champs(champ_list, mappos, vid, max_frame=9000)
vid.release()
cv2.destroyAllWindows()
################################################