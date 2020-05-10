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
                             DictProperty, ListProperty)
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
    bbox = DictProperty({'minx': float('inf'), 'miny': float('inf'),
                         'maxx': float('-inf'), 'maxy': float('-inf')})
    width = NumericProperty(0)
    height = NumericProperty(0)

    def __init__(self, touch, **kwargs):
        # The color is applied to all canvas items of this gesture
        self.color = kwargs.pop('color', [0., 0., 0.])

        super().__init__(**kwargs)
        # This is the touch.uid of the oldest touch represented
        self.id = str(touch.uid)
        # Key is touch.uid; value is a kivy.graphics.Line(); it's used even
        # if line_width is 0 (i.e. not actually drawn anywhere)
        # TODO: for our new use, this will just contain one stroke.
        self._strokes = {}
        # Make sure the bbox is up to date with the first touch position
        self.update_bbox(touch)

    def get_vectors(self, **kwargs):
        '''Return stroke vectors.'''
        vecs = []
        for tuid, l in self._strokes.items():
            vecs.append(list(zip(l.points[::2], l.points[1::2])))
        return vecs

    def handles(self, touch):
        '''Returns True if this container handles the given touch'''
        return str(touch.uid) in self._strokes

    def update_bbox(self, touch):
        '''Update gesture bbox from a touch coordinate'''
        x, y = touch.x, touch.y
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
    color = ListProperty([0., 0., 0.])
    draw_bbox = BooleanProperty(True)
    bbox_alpha = NumericProperty(0.1)
    erase_threshold = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # A list of StrokeContainer objects (all strokes on the surface)
        self._strokes = []
        self.undo_history = []
        self.redo_history = []
        self.register_event_type('on_canvas_change')
        self._mode = "write"

    def get_vectors(self):
        vecs = []
        for stroke in self._strokes:
            vecs += stroke.get_vectors()
        return vecs

    def clear(self):
        # TODO: allow bulk undoing a clear
        for stroke in self._strokes:
            self.redo_history.append(stroke)
        self._strokes = []
        self.canvas.clear()
        self.dispatch('on_canvas_change')

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

    def on_touch_move(self, touch):
        "When a touch moves, we add a point to the line on the canvas so the path is updated."
        if touch.grab_current is not self:
            return
        if not self.collide_point(touch.x, touch.y):
            return

        if self._mode == "write":
            # Retrieve the StrokeContainer object that handles this touch.
            s = self.get_stroke(touch)
            s.update_bbox(touch)
            # Add the new point to gesture stroke list and update the canvas line
            s._strokes[str(touch.uid)].points += (touch.x, touch.y)

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

        if self._mode == "write":
            s = self.get_stroke(touch)
            self.clear_redo_history()
            self.undo_history.append(s)
        self.dispatch('on_canvas_change')

    def erase(self, x, y):
        for idx, stroke in enumerate(self._strokes):
            bb = [stroke.bbox[k] for k in ('minx', 'miny', 'maxx', 'maxy')]
            if not util.is_bbox_intersecting_helper(bb, x, y):
                continue
            dist = np.min(np.linalg.norm(np.array(stroke.get_vectors()) - np.array([x, y])[None, :], axis=0))
            if dist < self.erase_threshold:
                self.canvas.remove_group(stroke.id)
                # TODO: put it on undo history, once we add 're-adding' ability to undo.
                self.redo_history.append(stroke)
                del self._strokes[idx]

# -----------------------------------------------------------------------------
# Stroke related methods
# -----------------------------------------------------------------------------
    def init_stroke(self, touch):
        '''Create a new gesture from touch, i.e. it's the first on
        surface, or was not close enough to any existing gesture (yet)'''
        col = self.color
        g = StrokeContainer(touch, color=col)

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

        self._strokes.append(g)

        # Old init_stroke here:
        points = [touch.x, touch.y]
        col = g.color

        new_line = Line(
            points=points,
            width=self.line_width,
            group=g.id)
        g._strokes[str(touch.uid)] = new_line

        if self.line_width:
            self.canvas.add(Color(col[0], col[1], col[2], mode='rgb', group=g.id))
            self.canvas.add(new_line)

        # Update the bbox in case; this will normally be done in on_touch_move,
        # but we want to update it also for a single press, force that here:
        g.update_bbox(touch)
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

        idx = self._strokes.index(g)
        del self._strokes[idx]
        self.dispatch('on_canvas_change')
        return gesture_id

    def redo(self):
        if len(self.redo_history) == 0:
            return

        g = self.redo_history.pop()
        self.undo_history.append(g)

        self._strokes.append(g)
        col = g.color
        for (_, line) in g._strokes.items():
            self.canvas.add(Color(col[0], col[1], col[2], mode='rgb', group=g.id))
            self.canvas.add(line)
        self.dispatch('on_canvas_change')

    def clear_history(self):
        self.undo_history = []
        self.redo_history = []

    def clear_redo_history(self):
        self.redo_history = []

    def get_stroke(self, touch):
        '''Returns StrokeContainer associated with given touch'''
        for g in self._strokes:
            if g.handles(touch):
                return g
        raise Exception('get_stroke() failed to identify ' + str(touch.uid))

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