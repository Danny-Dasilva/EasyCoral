from .svg import *
import collections
import time

class DetectConvert():    
    def convert(data):
        CSS_STYLES = str(CssStyle({'.back': Style(fill='black',
                                                  stroke='black',
                                                  stroke_width='0.5em'),
                               '.bbox': Style(fill_opacity=0.0,
                                                  stroke_width='0.1em')}))

        BBox = collections.namedtuple('BBox', ('x', 'y', 'w', 'h'))
        BBox.area = lambda self: self.w * self.h
        BBox.scale = lambda self, sx, sy: BBox(x=self.x * sx, y=self.y * sy, w=self.w * sx, h=self.h * sy)
        BBox.__str__ = lambda self: 'BBox(x=%.2f y=%.2f w=%.2f h=%.2f)' % self

        Object = collections.namedtuple('Object', ('label', 'score', 'bbox'))
        Object.__str__ = lambda self: 'Object(label=%s, score=%.2f, %s)' % self

        EMPTY_SVG = str(Svg())

        if(data == []):
            return EMPTY_SVG
        
        #fps_counter  = Utils.avg_fps_counter(30)
        #inference_rate = next(fps_counter)

        inference_time = data[0]["inference_time"]

        x0, y0, width, height = (0, 80, 640, 480)
        font_size = 0.03 * height

        defs = Defs()
        defs += CSS_STYLES
        doc = Svg(width=width, height=height, viewBox='%s %s %s %s' % (0, 80, 640, 480), font_size=font_size, font_family='monospace', font_weight=500)
        doc += defs

        objs = []

        for i in range(len(data)):
            x0, y0, x1, y1 = data[i]["box"]
            objs.append(Object(label=data[i]["label"], score=data[i]["score"], bbox=BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)))
        
        for obj in objs:
            percent = int(100 * obj.score)
            if obj.label:
                caption = '%d%% %s' % (percent, obj.label)
            else:
                caption = '%d%%' % percent
            size = (640, 640)
            x, y, w, h = obj.bbox.scale(*size)
            color = 'white'

            d_w = w / 4
            w_empty = w / 2
            both =  (w /4) + (h /4)
            d_h = (h /4)
            h_empty = h/2

            dash_array = f'{d_w}, {w_empty}, {both}, {h_empty}, {both}, {w_empty}, {both}, {h_empty}, {d_h}'
            doc += Rect(x=x, y=y, width=w, height=h, style='stroke:%s' % color, stroke_dasharray=dash_array, _class='bbox')
            doc += Rect(x=x, y=y+h, width='%sem' % str(0.6 * (len(caption) + 1)), height='1.2em', fill=color)
            t = Text(x=x, y=y+h, fill='black')
            t += TSpan(caption, dy='1em')
            doc += t
        
        ox = x0 + 20
        oy1, oy2 = y0 + 20 + font_size, y0 + height - 20

        lines = [
            'Objects: %d' % len(objs),
            'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time)
        ]

        for i, line in enumerate(reversed(lines)):
            y = oy2 - i * 1.7 * font_size
            doc += Rect(x=0, y=0, width='%sem' % str(0.6 * (len(line) + 1)), height='1em', transform='translate(%s, %s) scale(1,-1)' % (ox, y), _class='back')
            doc += Text(line, x=ox, y=y, fill='white')
        
        return str(doc)

class ClassifyConvert():    
    def convert(data):
        CSS_STYLES = str(CssStyle({'.back': Style(fill='black',
                                                  stroke='black',
                                                  stroke_width='0.5em')}))
        
        EMPTY_SVG = str(Svg())

        if(data == []):
            return EMPTY_SVG

        inference_time = data[0]["inference_time"]
        
        x0, y0, width, height = (0, 80, 640, 480)
        font_size = 0.03 * height

        ox1, ox2 = x0 + 20, x0 + width - 20
        oy1, oy2 = y0 + 20 + font_size, y0 + height - 20

        defs = Defs()
        defs += CSS_STYLES
        doc = Svg(width=width, height=height, viewBox='%s %s %s %s' % (0, 80, 640, 480), font_size=font_size, font_family='monospace', font_weight=500)
        doc += defs

        for i in range(len(data)):
            lines = ['%s (%.2f)' % (data[i]["label"], data[i]["score"])]
        
        for i, line in enumerate(lines):
            y = oy2 - i * 1.7 * font_size
            doc += Rect(x=0, y=0, width='%sem' % str(0.6 * (len(line) + 1)), height='1em', transform='translate(%s, %s) scale(-1,-1)' % (ox2, y), _class='back')
            doc += Text(line, text_anchor='end', x=ox2, y=y, fill='white')

        lines = [
            'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time)
        ]

        for i, line in enumerate(reversed(lines)):
            y = oy2 - i * 1.7 * font_size
            doc += Rect(x=0, y=0, width='%sem' % str(0.6 * (len(line) + 1)), height='1em', transform='translate(%s, %s) scale(1,-1)' % (ox1, y), _class='back')
            doc += Text(line, x=ox1, y=y, fill='white')

        return str(doc)
        
class Utils:
    def avg_fps_counter(window_size):
        window = collections.deque(maxlen=window_size)
        prev = time.monotonic()
        yield 0.0  # First fps value.
        while True:
            curr = time.monotonic()
            window.append(curr - prev)
            prev = curr
            yield len(window) / sum(window)