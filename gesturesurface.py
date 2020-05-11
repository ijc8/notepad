'''
Gesture Surface
===============

.. versionadded::
    1.9.0

.. warning::

    This is experimental and subject to change as long as this warning notice
    is present.

See :file:`kivy/examples/demo/multistroke/main.py` for a complete application
example.
'''
__all__ = ('StrokeSurface', 'StrokeContainer')

from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
                             DictProperty)
from colorsys import hsv_to_rgb

import numpy as np

import util


# Let's group gestures at a later time.
# TODO: Specialize so this really only contains one stroke.
class StrokeContainer(EventDispatcher):
    '''Container object that stores information about a stroke.

    :Arguments:
        `touch`
            Touch object (as received by on_touch_down) used to initialize
            the gesture container. Required.

        `bbox`
            Dictionary with keys minx, miny, maxx, maxy. Represents the size
            of the stroke bounding box.

        `width`
            Represents the width of the stroke.

        `height`
            Represents the height of the stroke.
    '''
    def __init__(self, line, **kwargs):
        # The color is applied to all canvas items of this gesture
        self.color = kwargs.pop('color', [0., 0., 0.])
        super().__init__(**kwargs)
        self.width = 0
        self.height = 0
        self.bbox = {'minx': float('inf'), 'miny': float('inf'),
                     'maxx': float('-inf'), 'maxy': float('-inf')}

        self.id = line.group
        self._stroke = line

        # Make sure the bbox is up to date with the first touch position
        for point in zip(line.points[::2], line.points[1::2]):
            self.update_bbox(*point)

    def get_points(self, **kwargs):
        "Return stroke points."
        return list(zip(self._stroke.points[::2], self._stroke.points[1::2]))

    def update_bbox(self, x, y):
        '''Update gesture bbox from a touch coordinate'''
        bb = self.bbox
        if x < bb['minx']:
            bb['minx'] = x
        if y < bb['miny']:
            bb['miny'] = y
        if x > bb['maxx']:
            bb['maxx'] = x
        if y > bb['maxy']:
            bb['maxy'] = y
        self.width = bb['maxx'] - bb['minx']
        self.height = bb['maxy'] - bb['miny']

    def serialize(self):
        stroke = {
            'points': self._stroke.points,
            'width': self._stroke.width,
            'group': self._stroke.group,
        }

        return {
            'color': self.color,
            'bbox': self.bbox,
            'width': self.width,
            'height': self.height,
            'stroke': stroke,
        }

    @staticmethod
    def deserialize(obj):
        stroke = obj['stroke']

        line = Line(
            points=stroke['points'],
            width=stroke['width'],
            group=util.loaded_id(stroke['group']),
            # touch.uid starts from 1,2,...
            # this makes sure that it doesn't collide with loaded data
        )
        s = StrokeContainer(line)
        s.color = obj['color']
        s.bbox = obj['bbox']
        s.width = obj['width']
        s.height = obj['height']
        return s

class StrokeSurface(FloatLayout):
    '''Drawing surface for strokes.

    :Properties:
        `color`
            Color used to draw the gesture, in RGB.

        `line_width`
            Line width used for tracing touches on the surface.

        `draw_bbox`
            Set to True if you want to draw bounding box behind gestures.

    :Events:
        `on_canvas_change`
            Fired when the canvas is modified, by the user writing or erasing strokes.
    '''

    line_width = NumericProperty(2)

    draw_bbox = BooleanProperty(True)
    bbox_alpha = NumericProperty(0.1)
    erase_threshold = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # A map of Touch ID to StrokeContainer objects (all strokes on the surface)
        self._strokes = {}
        self.undo_history = []
        self.redo_history = []
        self.color = [0., 0., 0.]
        self.register_event_type('on_canvas_change')
        self._mode = "write"
        self.artificial_id = 0

    def get_strokes(self):
        vecs = []
        for stroke in self._strokes.values():
            vecs.append((stroke.id, stroke.get_points()))
        return vecs

    def clear(self):
        # TODO: allow bulk undoing a clear
        for stroke in self._strokes.values():
            self.redo_history.append(stroke)
        self._strokes = {}
        self.canvas.clear()
        self.dispatch('on_canvas_change')

    def clear_history(self):
        self.undo_history = []
        self.redo_history = []

    def serialize(self):
        strokes = {}
        for id, stroke in self._strokes.items():
            strokes[id] = stroke.serialize()

        return {
            'strokes': strokes,
            'artificial_id': self.artificial_id,
            'color': self.color,
        }

    def populate(self, obj):
        self.undo_history = []
        self.redo_history = []

        self.artificial_id = obj['artificial_id']
        self.color = obj['color']

        strokes = obj['strokes']
        self._strokes = {}
        for old_id, serialized_stroke_obj in strokes.items():
            # touch.uid starts from 1,2,...
            # this makes sure that it doesn't collide with loaded data
            id = util.loaded_id(old_id)
            self._strokes[id] = StrokeContainer.deserialize(serialized_stroke_obj)

    def redraw_all(self):
        for id, stroke in self._strokes.items():
            col = stroke.color
            self.canvas.add(Color(col[0], col[1], col[2], mode='rgb', group=id))
            self.canvas.add(stroke._stroke)


# -----------------------------------------------------------------------------
# Touch Events
# -----------------------------------------------------------------------------
    def on_touch_down(self, touch, mode):
        "When a new touch is registered, we start a new stroke for it."
        # If the touch originates outside the surface, ignore it.
        if not self.collide_point(touch.x, touch.y):
            return
        touch.grab(self)
        self._mode = mode
        if mode == "write":
            self.init_stroke(touch)
        else:
            self.erase(touch.x, touch.y)
        return True

    def get_id(self, touch):
        return str(touch.uid)

    def on_touch_move(self, touch):
        "When a touch moves, we add a point to the line on the canvas so the path is updated."
        if touch.grab_current is not self:
            return
        if not self.collide_point(touch.x, touch.y):
            return

        if self._mode == "write":
            # Retrieve the StrokeContainer object that handles this touch.
            s = self._strokes[self.get_id(touch)]
            s.update_bbox(touch.x, touch.y)
            # Add the new point to gesture stroke list and update the canvas line
            s._stroke.points += (touch.x, touch.y)

            # Draw the gesture bounding box; if it is a single press that
            # does not trigger a move event, we would miss it otherwise.
            if self.draw_bbox:
                self._update_canvas_bbox(s)
        else:
            self.erase(touch.x, touch.y)

        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return
        touch.ungrab(self)

        id = self.get_id(touch)
        if self._mode == "write":
            s = self._strokes[id]
            if len(s._stroke.points) < 2 or not (s.width > 5 or s.height > 5):
                self.canvas.remove_group(id)
                del self._strokes[id]
            else:
                self.clear_redo_history()
                self.undo_history.append(s)
        self.dispatch('on_canvas_change')

    def add_stroke(self, points):
        "Add a new stroke to the Surface (as if the user drew it)."
        id = f'artificial touch {self.artificial_id}'
        self.artificial_id += 1
        line = Line(points=points.flatten().tolist(),
                    width=self.line_width,
                    group=id)
        s = StrokeContainer(line)
        self._strokes[id] = s
        col = s.color
        self.canvas.add(Color(col[0], col[1], col[2], mode='rgb', group=id))
        self.canvas.add(line)
        self.dispatch('on_canvas_change')

    def erase(self, x, y):
        dead = []
        for id, stroke in self._strokes.items():
            bb = [stroke.bbox[k] for k in ('minx', 'miny', 'maxx', 'maxy')]
            if not util.is_bbox_intersecting_helper(bb, x, y):
                continue
            dist = np.min(np.linalg.norm(np.array(stroke.get_points()) - np.array([x, y])[None, :], axis=1))
            if dist < self.erase_threshold:
                self.canvas.remove_group(id)
                # TODO: put it on undo history, once we add 're-adding' ability to undo.
                self.redo_history.append(stroke)
                dead.append(id)

        for id in dead:
            del self._strokes[id]

# -----------------------------------------------------------------------------
# Stroke related methods
# -----------------------------------------------------------------------------
    def init_stroke(self, touch):
        '''Create a new gesture from touch, i.e. it's the first on
        surface, or was not close enough to any existing gesture (yet)'''
        col = self.color
        id = self.get_id(touch)
        points = [touch.x, touch.y]
        line = Line(points=points,
                    width=self.line_width,
                    group=id)

        g = StrokeContainer(line, color=col)
        self._strokes[id] = g

        # Create the bounding box Rectangle for the gesture
        if self.draw_bbox:
            bb = g.bbox
            with self.canvas:
                Color(col[0], col[1], col[2], self.bbox_alpha, mode='rgba',
                      group=g.id)

                g._bbrect = Rectangle(
                    group=g.id,
                    pos=(bb['minx'], bb['miny']),
                    size=(bb['maxx'] - bb['minx'],
                          bb['maxy'] - bb['miny']))


        col = g.color

        if self.line_width:
            self.canvas.add(Color(col[0], col[1], col[2], mode='rgb', group=g.id))
            self.canvas.add(line)

        if self.draw_bbox:
            self._update_canvas_bbox(g)

        return g

    def undo(self):
        if len(self.undo_history) == 0:
            return

        g = self.undo_history.pop()
        gesture_id = g.id
        self.canvas.remove_group(gesture_id)
        self.redo_history.append(g)

        del self._strokes[g.id]
        self.dispatch('on_canvas_change')
        return gesture_id

    def redo(self):
        if len(self.redo_history) == 0:
            return

        g = self.redo_history.pop()
        self.undo_history.append(g)

        self._strokes[g.id] = g
        col = g.color
        self.canvas.add(Color(col[0], col[1], col[2], mode='rgb', group=g.id))
        self.canvas.add(g._stroke)
        self.dispatch('on_canvas_change')

    def clear_history(self):
        self.undo_history = []
        self.redo_history = []

    def clear_redo_history(self):
        self.redo_history = []

    def _update_canvas_bbox(self, g):
        # If draw_bbox is changed while two gestures are active,
        # we might not have a bbrect member
        if not hasattr(g, '_bbrect'):
            return

        bb = g.bbox
        g._bbrect.pos = (bb['minx'], bb['miny'])
        g._bbrect.size = (bb['maxx'] - bb['minx'],
                          bb['maxy'] - bb['miny'])

    def on_canvas_change(self, *l):
        pass
