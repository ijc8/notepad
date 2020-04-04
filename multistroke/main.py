'''
Multistroke Recognition Database Demonstration
==============================================

This application records gestures and attempts to match them. You should
see a black drawing surface with some buttons across the bottom. As you
make a gesture on the drawing surface, the gesture will be added to
the history and a match will be attempted. If you go to the history tab,
name the gesture, and add it to the database, then similar gestures in the
future will be recognized. You can load and save databases of gestures
in .kg files.

This demonstration code spans many files, with this being the primary file.
The information pop-up ('No match') comes from the file helpers.py.
The history pane is managed in the file historymanager.py and described
in the file historymanager.kv. The database pane and storage is managed in
the file gesturedatabase.py and the described in the file gesturedatabase.kv.
The general logic of the sliders and buttons are in the file
settings.py and described in settings.kv. but the actual settings pane is
described in the file multistroke.kv and managed from this file.

'''
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.gesturesurface import GestureSurface
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.label import Label
from kivy.multistroke import Recognizer

from kivy.graphics import Ellipse, Color, Line
import numpy as np

# Local libraries
from historymanager import GestureHistoryManager
from gesturedatabase import GestureDatabase
from settings import MultistrokeSettingsContainer

import util

class MainMenu(GridLayout):
    pass


class MultistrokeAppSettings(MultistrokeSettingsContainer):
    pass


class NotePadSurface(GestureSurface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lines = []
        self.spacing = 1/16
        with self.canvas.before:
            Color(0, 0, 0, 1)
            for i in range(0, 5):
                self.lines.append(Line(points=self.get_points(i)))

    def get_height(self, line_number):
        return self.size[1] * (1/2 - (line_number - 2) * self.spacing)

    def get_points(self, line_number):
        height = self.get_height(line_number)
        return [0, height, self.size[0], height]

    def on_size(self, foo, bar):
        for i, line in enumerate(self.lines):
            line.points = self.get_points(i)

    def _cleanup(self, dt):
        print('jk')


class Note:
    def __init__(self, pitch, duration, gesture):
        self.pitch = pitch
        self.duration = duration
        self.gesture = gesture


class MultistrokeApp(App):

    def goto_database_screen(self, *l):
        self.database.import_gdb()
        self.manager.current = 'database'

    def handle_gesture_cleanup(self, surface, g, *l):
        if hasattr(g, '_result_label'):
            surface.remove_widget(g._result_label)

    def handle_gesture_discard(self, surface, g, *l):
        # Don't bother creating Label if it's not going to be drawn
        if surface.draw_timeout == 0:
            return

        text = '[b]Discarded:[/b] Not enough input'
        g._result_label = Label(text=text, markup=True, size_hint=(None, None),
                                center=(g.bbox['minx'], g.bbox['miny']))
        self.surface.add_widget(g._result_label)

    def handle_gesture_complete(self, surface, g, *l):
        result = self.recognizer.recognize(g.get_vectors())
        # TODO: temp visualization
        points = np.array(sum(g.get_vectors(), []))
        center = points.mean(axis=0)
        points = points[util.reject_outliers(points[:, 1], verbose=True)]
        new_center = points.mean(axis=0)
        radius = 5
        with self.surface.canvas:
            Color(1, 1, 1, 1)
            Ellipse(pos=center - radius, size=(radius * 2, radius * 2))
            Color(0, 0, 0, 1)
            Ellipse(pos=new_center - radius, size=(radius * 2, radius * 2))
        # end tmp
        result._gesture_obj = g
        result.bind(on_complete=self.handle_recognize_complete)

    def handle_recognize_complete(self, result, *l):
        self.history.add_recognizer_result(result)

        # Don't bother creating Label if it's not going to be drawn
        if self.surface.draw_timeout == 0:
            return

        best = result.best
        if best['name'] is None:
            text = '[b]No match[/b]'
        else:
            text = 'Name: [b]%s[/b]\nScore: [b]%f[/b]\nDistance: [b]%f[/b]' % (
                   best['name'], best['score'], best['dist'])

        text = f'[color=#000000]{text}[/color]'
        g = result._gesture_obj
        g._result_label = Label(text=text, markup=True, size_hint=(None, None),
                                center=(g.bbox['minx'], g.bbox['miny']))
        if best['name'] and best['name'].endswith('note'):
            print("dude it's a note")
            points = np.array(sum(g.get_vectors(), []))
            points = points[util.reject_outliers(points[:, 1])]
            note_height = points[:, 1].mean()
            pitches = 'E F G A B C D e f'.split()[::-1]
            pitch = pitches[min(range(0, 9), key=lambda i: np.abs(self.surface.get_height(i / 2) - note_height))]
            durations = {'quarternote': 1/4, 'halfnote': 1/2, 'wholenote': 1}
            self.notes.append(Note(pitch, durations[best['name']], g))
            group = list(self.surface.canvas.get_group(g.id))
            for i0, i1 in zip(group, group[2:]):
                if isinstance(i0, Color) and isinstance(i1, Line):
                    i0.rgba = (0, 0, 0, 1)
            print([(note.pitch, int(note.duration * 4)) for note in self.notes])
        self.surface.add_widget(g._result_label)

    def build(self):
        # TODO: __init__
        self.notes = []
        # Setting NoTransition breaks the "history" screen! Possibly related
        # to some inexplicable rendering bugs on my particular system
        self.manager = ScreenManager(transition=SlideTransition(
                                     duration=.15))
        self.recognizer = Recognizer()

        # Setup the GestureSurface and bindings to our Recognizer
        surface = NotePadSurface(line_width=2, draw_bbox=True,
                                 use_random_color=True)
        surface_screen = Screen(name='surface')
        surface_screen.add_widget(surface)
        self.manager.add_widget(surface_screen)

        surface.bind(on_gesture_discard=self.handle_gesture_discard)
        surface.bind(on_gesture_complete=self.handle_gesture_complete)
        surface.bind(on_gesture_cleanup=self.handle_gesture_cleanup)
        self.surface = surface

        # History is the list of gestures drawn on the surface
        history = GestureHistoryManager()
        history_screen = Screen(name='history')
        history_screen.add_widget(history)
        self.history = history
        self.manager.add_widget(history_screen)

        # Database is the list of gesture templates in Recognizer
        database = GestureDatabase(recognizer=self.recognizer)
        database_screen = Screen(name='database')
        database_screen.add_widget(database)
        self.database = database
        self.manager.add_widget(database_screen)

        # Settings screen
        app_settings = MultistrokeAppSettings()
        ids = app_settings.ids

        ids.max_strokes.bind(value=surface.setter('max_strokes'))
        ids.temporal_win.bind(value=surface.setter('temporal_window'))
        ids.timeout.bind(value=surface.setter('draw_timeout'))
        ids.line_width.bind(value=surface.setter('line_width'))
        ids.draw_bbox.bind(value=surface.setter('draw_bbox'))
        ids.use_random_color.bind(value=surface.setter('use_random_color'))

        settings_screen = Screen(name='settings')
        settings_screen.add_widget(app_settings)
        self.manager.add_widget(settings_screen)

        # Wrap in a gridlayout so the main menu is always visible
        layout = GridLayout(cols=1)
        layout.add_widget(self.manager)
        layout.add_widget(MainMenu())
        return layout


if __name__ in ('__main__', '__android__'):
    MultistrokeApp().run()
