### live game detection with voice alarm when enemy jungler ganks.

import cv2, pafy
import numpy as np
from PIL import ImageGrab, Image, ImageTk
import json
import pyttsx3 as t2s
import tkinter as tk
import win32gui as wui
import win32api as wap
import win32con as won
from time import sleep

root = tk.Tk()

champ_names = {'Jarvan IV':['Jarvan','J4'],
               'Miss Fortune':['Mf'],
               'Aurelion Sol':['As','Asol'],
               'Blitzcrank':['Blitz'],
               'Caitlyn':['Cait'],
               'Cassiopeia':['Cass'],
               "Cho'Gath":['Cho','Chogath'],
               'Dr.Mundo':['Mundo'],
               'Evelynn':['Eve'],
               'Fiddlesticks':['Fiddle'],
               'Gangplank':['Gp'],
               'Heimerdinger':['Heimer'],
               "Kai'sa":['Kaisa'],
               'Kassadin':['Kass'],
               'Katarina':['Kata'],
               "Kha'Zix":['Kha','Khazix',"Kha'zix"],
               "Kog'Maw":['Kog','Kogmaw',"Kog'maw"],
               "LeBlanc":['Lb','Leblanc'],
               'Lee Sin':['Lee','Lee sin','Leesin'],
               'Lissandra':['Liss'],
               'Master Yi':['Master yi','Yi'],
               'Mordekaiser':['Morde'],
               'Nautilus':['Nauti','Naut'],
               'Nidalee':['Nid','Nidali'],
               'Nocturne':['Noc','Noct'],
               'Pantheon':['Panth'],
               "Rek'Sai":['Reksai',"Rek'sai"],
               'Sejuani':['Sej'],
               'Shyvana':['Shyv'],
               'Tahm Kench':['Tahm','Tahmkench','Tahm kench'],
               'Tryndamere':['Trynda'],
               'Twisted Fate':['Tf'],
               "Vel'Koz":['Velkoz','Velcoz',"Vel'Coz",'Vel'],
               'Vladimir':['Vlad'],
               'Volibear':['Voli'],
               'Xin Zhao':['Xin','Zhao','Xinzhao'],
               'Yorick':['Yorik','Yoric'],
               'Lucian':['Luci'],
               'Ezreal':['Ez'],
               'Ryze':['Rize'],
               'Neeko':['Niko','Neeco','Nico'],
               'Tristana':['Trist'],
               'Thresh':['Th'],
               'Orianna':['Ori','Oriana'],
               'Morgana':['Morg'],
               'Gragas':['Grag']}

def simplify(name):
    name = name.capitalize()
    for k, v in champ_names.items():
        if name in v:
            return k
    return name

## to simplify color conversion
def grayify(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## load video from youtube
def load_yt(url):
    print(f'start loading from {url}')
    vpafy = pafy.new(url)
    best = vpafy.getbest("any", False) # we don't need audio there.
    print('title:',best.title)
    print('filesize',best.get_filesize())
    print('quality',best.resolution)
    return best.url

class Tracker:
    def __init__(self, name):
        self.name = simplify(name)
        self._filename = f'champion_icons/{self.name}.jpg'
        self._icon = cv2.imread(self._filename)
        self._icon = cv2.resize(self._icon, (24, 24), interpolation = cv2.INTER_AREA)
        self._icon_gray = cv2.cvtColor(self._icon, cv2.COLOR_BGR2GRAY)
        self.path = []
        self.counter = 0
        self.last_seen = (0,0)
        self.unseen_count = 0
        
        
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
            self.last_seen = self.position
        else:
            self.unseen_count += 1
        if self.unseen_count >= 3:
            self.unseen_count = 0
            self.last_seen = [-100, -100]
        return self.position
        
class MultiTracker:
    def __init__(self, targets):
        self._trackers = [ Tracker(i) for i in targets]
    
    def __getitem__(self, key):
        for i in self._trackers:
            if i.name == key:
                return i
        raise ValueError
    
    def track(self, image):
        result = dict()
        for i in self._trackers:
            pos = i.track(image)
            result[i.name] = pos
        return result
    
def detect_minimap(video = None):
    
    minimap = cv2.imread('minimap1.png')    
    minimap = grayify(minimap)
    minimap_edges_og = cv2.Canny(minimap, 25, 50)
    map_found = False
    mappos = (0,0,0,0)
    while(not map_found):
        if not video:
            frame = np.array(ImageGrab.grab())
        else:
            _, frame = video.read()
        
        mid_line = int(0.65 * frame.shape[0])
        bottom = frame[int(mid_line):,:,:]
        bottom = grayify(bottom)
        edges = cv2.Canny(bottom, 25, 50)
        minimap_edges = minimap_edges_og.copy()
        temp = None
        maxres = None
        probs = dict()
        if(minimap_edges.shape[0] > edges.shape[0]):
            minimap_edges = cv2.resize(minimap_edges_og, (edges.shape[0], edges.shape[0]))
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
            mappos = probs[maxres]
            map_found = True
        print('minimap found')
        return mappos



def track_champs(champ_list, mappos, video = None, save_name = None, max_frame = None, voice_alarm = False):
    first_frame = True
    trackers = MultiTracker(champ_list)
    counter = 0
    path = dict()
    mapsize = mappos[2] - mappos[0]
    
    champ_list = [ simplify(i) for i in champ_list]
    allies = champ_list[:5]
    topmidbot = [allies[0],allies[2],allies[3]]
    enemies = champ_list[5:]
    enemy_jung = enemies[1]
    print(enemy_jung,'is the enemy jungler')
    
     #For icon
    root.geometry(f"{mappos[2]-mappos[0]}x{mappos[3]-mappos[1]}+{mappos[0]}+{mappos[1]}")
    image_label = tk.Label(root, bg='white',image=empty_image)
    image_label.pack()
    
    if save_name:
        vidfcc = cv2.VideoWriter_fourcc(*'XVID')
        vidwriter  = cv2.VideoWriter(save_name+'.mp4',vidfcc, 30.0, (int(screen_w), int(screen_h)))
        
    if voice_alarm:
        voice_engine = t2s.init()
        last_alarm = 0
    while(True):
        root.attributes('-topmost', 1)
        if not video:
            frame = np.array(ImageGrab.grab())
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print(frame.shape)
            
        else:
            _, frame = video.read()

#        frame_small = cv2.resize(frame, (1280, 720))
        if np.all(frame) == None:
            break
        
        maparea = frame[mappos[1]:mappos[3],mappos[0]:mappos[2]]
        maparea = cv2.resize(maparea, (240, 240))
        showcase = np.ones_like(maparea) * 255
        showcase_rgba = np.zeros([mapsize, mapsize, 4], dtype = np.uint8)
        maparea = grayify(maparea)

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
        alarm_text = ''
        champ_radius = 12 / 240 * mapsize
        print(int(champ_radius))
        for k, v in poses.items():
            if v:
                if k == enemy_jung and voice_alarm:
                    for ally in topmidbot:
                        distance = np.linalg.norm(np.array(trackers[ally].last_seen) - np.array(v))
                        if distance <= 30:
                            alarm_text = f'{enemy_jung} is ganking {ally}'
                            print(alarm_text)
#                            voice_engine.say(f'{enemy_jung} is ganking {ally}')
                v = (np.array(v) / 240 * mapsize).astype(np.int)

                cv2.putText(showcase_rgba, k, (int(v[0]-1.5*champ_radius),int(v[1]+2 * champ_radius)) , font, 0.3, (254,255,255,255), 1, cv2.LINE_8)
                cv2.circle(showcase_rgba, tuple(v), int(champ_radius), (0, 255,255,255), 2, cv2.LINE_8)
                val = tuple([counter]+list(v)) # (frame, x_pos, y_pos)
                if k not in path.keys():
                    path[k] = [val]
                else:
                    path[k].append(val)
            else:
                v = trackers[k].last_seen
                v = (np.array(v) / 240 * mapsize).astype(np.int)
                cv2.putText(showcase_rgba, k, (int(v[0]-1.5*champ_radius),int(v[1]+2 * champ_radius)), font, 0.3, (254,255,255,255), 1, cv2.LINE_8)
                cv2.circle(showcase_rgba, tuple(v), int(champ_radius), (0, 255, 255,255), 2, cv2.LINE_8)

        if voice_alarm and alarm_text:
            if counter - last_alarm >= 90:
                last_alarm = counter
                voice_engine.say(alarm_text)

        if len(showcase.shape) == 2:
            showcase = cv2.cvtColor(showcase, cv2.COLOR_GRAY2BGR)
        
        showcase_rgba = cv2.cvtColor(showcase_rgba, cv2.COLOR_BGRA2RGBA)
        showcase_tk = ImageTk.PhotoImage(image=Image.fromarray(showcase_rgba))
#        cv2.imshow('show', showcase_rgba)
        image_label.configure(image = showcase_tk)
        image_label.image = showcase_tk
#        cv2.imshow('show',showcase)
        
        prev_frame = maparea
        
#        if voice_alarm:
#            voice_engine.runAndWait()
        if save_name:
            save_frame = np.array(ImageGrab.grab())
            save_frame = cv2.cvtColor(save_frame, cv2.COLOR_RGB2BGR)
            vidwriter.write(save_frame)
        if max_frame:
            if counter >= max_frame:
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        root.update()
        counter += 1
        
    root.destroy()
    if save_name:
        with open(save_name+'.json','w') as f:
            dump = json.dumps(path)
            f.write(dump)
        vidwriter.release()
    return path

try:
    sleep(15)

# order is important!!! must be: ['ally top', 'jgl','mid','adc','sup', 'enemy top','jgl','mid','adc','sup']
    champ_list = ['Renekton','taliyah','lulu','vayne','blitz','cho','Lee Sin','ahri','lucian',"nami"]

    mappos = detect_minimap(vid)

    root.overrideredirect(1)
    root.attributes('-topmost', 1)
    root.wm_attributes('-transparentcolor','white')
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    empty_array = np.zeros((mappos[2]-mappos[0],mappos[2]-mappos[0],4), dtype=np.uint8)
    empty_image = ImageTk.PhotoImage(image=Image.fromarray(empty_array))
    
    path = track_champs(champ_list, mappos,video = None ,max_frame=30 * 60 * 10, voice_alarm = True)
    if vid:
        vid.release()
finally:
    cv2.destroyAllWindows()
    root.destroy()
