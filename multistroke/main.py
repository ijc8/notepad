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
# Built-in modules
import sys
sys.path += ['.', '..']

durations = {'eighth': 1/8, 'quarter': 1/4, 'half': 1/2, 'whole': 1}

# Kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.gesturesurface import GestureSurface
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.label import Label
from kivy.uix.scatter import ScatterPlane
from kivy.graphics import Ellipse, Color, Line

# Other external libraries
import numpy as np
import fluidsynth
import pyaudio
import wave
import pickle
import copy

from dollarpy import Recognizer, Template, Point

# Local libraries
from historymanager import GestureHistoryManager
from gesturedatabase import GestureDatabase
from settings import MultistrokeSettingsContainer
import util
import math
import transcribe


class MainMenu(GridLayout):
    pass


class MultistrokeAppSettings(MultistrokeSettingsContainer):
    pass


class NotePadSurface(GestureSurface):
    def on_kv_post(self, base_widget):
        super().on_kv_post(base_widget)
        # Draw lines here, because self.size isn't set yet in __init__().
        self.lines = []
        self.staff_spacing = self.size[1] / 12
        self.line_spacing = self.size[1] / 96
        with self.canvas.before:
            Color(0, 0, 0, 1)
            for staff in range(2):
                for line in range(5):
                    self.lines.append(Line(points=self.get_points(staff, line)))

    def get_height(self, staff_number, line_number):
        return self.size[1] - self.staff_spacing * (staff_number + 1) - self.line_spacing * (line_number - 2)

    def get_points(self, staff_number, line_number):
        height = self.get_height(staff_number, line_number)
        return [0, height, self.size[0], height]

    def on_touch_down(self, touch):
        if 'button' in touch.profile and touch.button != 'left':
            # Don't handle right or middle-clicks: let the ScatterPlane take them.
            return False
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if 'button' in touch.profile and touch.button != 'left':
            # Don't handle right or middle-clicks: let the ScatterPlane take them.
            return False
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if 'button' in touch.profile and touch.button != 'left':
            # Don't handle right or middle-clicks: let the ScatterPlane take them.
            return False
        return super().on_touch_up(touch)

    def get_note_gesture(self, value, pitch):
        gesture = []
        value = value / 4
        for name, duration in durations.items():
            if duration == value:
                filename = name + ('note' if pitch else 'rest')
                with open("ink/" + filename, "rb") as data_file:
                    gesture = pickle.load(data_file)
                return gesture
        return gesture

    def get_y_from_pitch(self, staff_number, pitch):
        # Special case: put rests in the middle of the staff.
        if pitch == 0:
            return self.get_height(staff_number, 2)
        pitches = [79, 77, 76, 74, 72, 71, 69, 67, 65, 64, 62]
        idx = pitches.index(pitch) - 1
        return self.get_height(staff_number, idx / 2)

    def draw_ink_based_on_melody(self, staff_number, x_start, melody):
        xs = []
        for note in melody:
            xs.append(x_start)
            x_next_start = self.draw_ink_based_on_note(staff_number, x_start, note)
            x_start = x_next_start
        return xs

    def align_note(self, note, points):
        "Normalize and translate notes so (0, 0) is where the note should be pinned on a staff."
        (_, value, pitch) = note
        mins, maxs = util.get_bounds(points)
        width, height = maxs - mins
        if pitch:
            # Normalize notes - size of notehead should equal line spacing.
            normalization_factor = self.line_spacing / width
            points = mins + (points - mins) * normalization_factor

            # Translate - set (0, 0) to center of notehead.
            points_without_outliers = points[util.reject_outliers(points[:, 1])]
            center = points_without_outliers.mean(axis=0)
            return points - center
        elif value == 4:
            # For whole rests, top should be aligned to y = 0.
            return points - points.mean(axis=0) - height / 2
        elif value == 2:
            # For half rests, bottom should be aligned to y = 0.
            return points - points.mean(axis=0) + height / 2
        else:
            return points - points.mean(axis=0)

    def draw_ink_based_on_note(self, staff_number, x_start, note):
        (_, value, pitch) = note
        gesture = self.get_note_gesture(value, pitch)
        points = np.array(sum(gesture, []))
        bar_size = 12 * self.line_spacing
        note_padding = (bar_size * (value / 4.0)) / 2
        points = self.align_note(note, points)
        new_center_x = x_start + note_padding
        points += (new_center_x, self.get_y_from_pitch(staff_number, pitch))

        with self.canvas.before:
            Color(1.0, 0.0, 0.0, mode='rgb')
            Line(points=points.flat, group='gesture', width=self.line_width)

        return new_center_x + note_padding

class NotePadContainer(ScatterPlane):
    # Don't steal events from the menu above this...
    def on_touch_down(self, touch):
        if not self.y < touch.y < self.top:
            return False
        super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if not self.y < touch.y < self.top:
            return False
        super().on_touch_up(touch)

    def on_touch_move(self, touch):
        if not self.y < touch.y < self.top:
            return False
        super().on_touch_move(touch)

class NotePadMenu(GridLayout):
    pass


class Note:
    def __init__(self, pitch, duration, x_pos):
        self.pitch = pitch
        self.duration = duration
        self.x_pos = x_pos

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
        dollarResult = self.recognizer.recognize(
            util.convert_to_dollar(g.get_vectors()))

        result = util.ResultWrapper(dollarResult)

        result._gesture_obj = g

        self.handle_recognize_complete(result)

    def add_to_history_for_undo_redo(self, gesture, note=None):
        self.redo_history = []
        self.undo_history.append((gesture.id, gesture.get_vectors(), note))

    def handle_recognize_complete(self, result, *l):
        self.history.add_recognizer_result(result)

        # Don't bother creating Label if it's not going to be drawn
        if self.surface.draw_timeout == 0:
            return

        best = result.best
        g = result._gesture_obj

        if best['name'] is None or best['name'] == 'trebleclef':
            # No match or ignored. Leave it onscreen in case it's for the user's benefit.
            group = list(self.surface.canvas.get_group(g.id))
            for i0, i1 in zip(group, group[2:]):
                if isinstance(i0, Color) and isinstance(i1, Line):
                    i0.rgba = (0, 0, 0, 1)
            g._cleanup_time = None
            self.add_to_history_for_undo_redo(g)
            return

        text = 'Name: [b]%s[/b]\nScore: [b]%f[/b]\nDistance: [b]%f[/b]' % (
                best['name'], best['score'], best['dist'])

        text = f'[color=#000000]{text}[/color]'
        g._result_label = Label(text=text, markup=True, size_hint=(None, None),
                                center=(g.bbox['minx'], g.bbox['miny']))

        if best['name'].endswith('note'):
            points = np.array(sum(g.get_vectors(), []))

            # For saving inks
            # with open("ink/eighthnote", "wb") as data_file:
            #    pickle.dump(g.get_vectors(), data_file)

            # TODO: temp visualization
            center = points.mean(axis=0)
            points = points[util.reject_outliers(points[:, 1])]
            new_center = points.mean(axis=0)
            radius = 5
            with self.surface.canvas:
                Color(1, 1, 1, 1)
                Ellipse(pos=center - radius, size=(radius * 2, radius * 2))
                Color(0, 0, 0, 1)
                Ellipse(pos=new_center - radius, size=(radius * 2, radius * 2))
            # end tmp

            note_height = points[:, 1].mean()
            x_pos = points[:, 0].mean()
            pitches = [64, 65, 67, 69, 71, 72, 74, 76, 77][::-1]
            pitch = pitches[min(range(0, 9), key=lambda i: np.abs(self.surface.get_height(0, i / 2) - note_height))]
            note = Note(pitch, durations[best['name'][:-4]], x_pos)
            self.notes.append(note)
            self.notes.sort(key=lambda note: note.x_pos)
            # Hacky way to change note color to black once it's registered.
            group = list(self.surface.canvas.get_group(g.id))
            for i0, i1 in zip(group, group[2:]):
                if isinstance(i0, Color) and isinstance(i1, Line):
                    i0.rgba = (0, 0, 0, 1)
            g._cleanup_time = None
            self.add_to_history_for_undo_redo(g, note)

        if best['name'].endswith('rest'):
            points = np.array(sum(g.get_vectors(), []))

            # For saving inks
            # with open("ink/eighthrest", "wb") as data_file:
            #    pickle.dump(g.get_vectors(), data_file)

            x_pos = points[:, 0].mean()
            note = Note(0, durations[best['name'][:-4]], x_pos)
            self.notes.append(note)
            self.notes.sort(key=lambda note: note.x_pos)
            # Hacky way to change rest color to black once it's registered.
            group = list(self.surface.canvas.get_group(g.id))
            for i0, i1 in zip(group, group[2:]):
                if isinstance(i0, Color) and isinstance(i1, Line):
                    i0.rgba = (0, 0, 0, 1)
            g._cleanup_time = None
            self.add_to_history_for_undo_redo(g, note)

        # Check is same as 'check' mark.
        if best['name'].endswith('play'):
            self.playback()

        # Loop sign is half circle sign.
        if best['name'].endswith('loop'):
            self.shouldLoop = True
            self.loop()

        # Stop sign is 'X' mark
        if best['name'].endswith('stop'):
            self.shouldLoop = False

        self.surface.add_widget(g._result_label)

    def loop(self):
        if not self.shouldLoop:
            return

        def loop_callback(time, event, seq, data):
            if not self.shouldLoop:
                return
            self.loop()

        t = self.playback()
        callbackID = self.seq.register_client(
            name="loop_callback",
            callback=loop_callback,
        )

        # Pause in between loops
        t += 1000
        self.seq.timer(int(t), dest=callbackID, absolute=False)

    def playback(self):
        t = 0
        for note in self.notes:
            t_duration = self.beats_to_ticks(note.duration)
            if note.pitch > 0:
                self.seq.note_on(time=int(t), absolute=False, channel=0, key=note.pitch, dest=self.synthID, velocity=80)
            t += t_duration
        return t

    def clear(self):
        self.undo_history = []
        self.redo_history = []
        self.notes = []
        self.surface._gestures = []
        self.surface.canvas.clear()
        return

    def undo(self):
        if len(self.undo_history) == 0:
            return
        group_id, vectors, note = self.undo_history[-1]
        self.undo_history.pop()
        self.surface.canvas.remove_group(group_id)
        note_copy = None
        if note:
            note_copy = copy.deepcopy(note)
            self.notes.remove(note)
            self.notes.sort(key=lambda note: note.x_pos)

        self.redo_history.append((group_id, vectors, note_copy))

    def redo(self):
        if len(self.redo_history) == 0:
            return

        group_id, vectors, note = self.redo_history[-1]
        self.redo_history.pop()

        vectors = np.array(vectors)
        with self.surface.canvas:
            Color(0, 0, 0, 1)
            Line(points=vectors.flat, group=group_id, width=2)

        self.notes.append(note)
        self.notes.sort(key=lambda note: note.x_pos)
        self.undo_history.append((group_id, vectors, note))


    def record(self, save=False):
        """Helper function. Plays one measure of beats and then records one measure of audio."""

        # Play four beeps to indicate tempo and key.
        for i in range(4):
            self.seq.note_on(time=self.beats_to_ticks(i + 1), absolute=False, channel=0, key=60, dest=self.synthID, velocity=80)

        sr = 44100
        frame_size = 1024
        # TODO: for now, locked to one measure of recording. figure out actual policy.
        # (10 beats to include the calibration measure + latency allowance on each side)
        length = 10 / (self.tempo / 60)  # seconds
        print("recording")
        stream = self.audio.open(format=pyaudio.paInt16, channels=1,
                                 rate=sr, input=True,
                                 frames_per_buffer=frame_size)
        print('latencies', stream.get_input_latency(), stream.get_output_latency())
        # for the moment we're assuming pyaudio's output latency is a good estimate of fluidsynth's...
        latency = stream.get_input_latency() + stream.get_output_latency()

        frames = []
        for i in range(0, int(sr / frame_size * length)):
            data = stream.read(frame_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        data = b''.join(frames)
        data = np.frombuffer(data, dtype=np.int16).astype(np.int)
        # Throw out the first five beats plus latency, and the last beat.
        start = (60 / self.tempo) * 5 + latency
        end = (60 / self.tempo) * 9 + latency
        data = data[int(start * sr):int(end * sr)]

        if save:
            outfile = 'recorded.wav'
            f = wave.open(outfile, 'wb')
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(data.astype(np.int16).tobytes())
            f.close()
            print(f'saved recording to {outfile}.')

        return data, sr

    def record_rhythm(self):
        audio, sr = self.record()
        rhythm = list(transcribe.extract_rhythm(audio, sr, self.tempo, verbose=True))
        print('rhythm', rhythm)
        # HACK for prototype demo
        melody = []
        for (start, end) in zip(rhythm, rhythm[1:] + [4]):
            melody.append((start, end - start, 64))
        print(melody)
        xs = self.surface.draw_ink_based_on_melody(0, 20, melody)
        self.notes += [Note(pitch, value, x) for (_, value, pitch), x in zip(melody, xs)]

    def record_melody(self):
        audio, sr = self.record()
        melody = transcribe.extract_melody(audio, sr, self.tempo, verbose=True)
        # For the demo, we're going to keep this in the treble clef: say, in a range of 60 to 84.
        # TODO: draw ledger lines
        melody = [(s, v, (p - 12) % 24 + 60 if p else 0) for s, v, p in melody]
        print(melody)
        xs = self.surface.draw_ink_based_on_melody(0, 20, melody)
        self.notes += [Note(pitch, value, x) for (_, value, pitch), x in zip(melody, xs)]
        print('melody', melody)

    def beats_to_ticks(self, beats):
        ticks = self.time_scale / (self.tempo / 60) * beats
        # TODO: temp for debugging
        assert(ticks == int(ticks), f'{beats} and {ticks}')
        return int(round(ticks))

    def build(self):
        # TODO: __init__?
        self.time_scale = 1000
        self.tempo = 120  # bpm
        self.audio = pyaudio.PyAudio()
        self.notes = []
        self.seq = fluidsynth.Sequencer(time_scale=self.time_scale)
        self.fs = fluidsynth.Synth()

        sfid = None
        if sys.platform == "darwin":
            self.fs.start('coreaudio')
            sfid = self.fs.sfload("/Library/Audio/Sounds/Banks/FluidR3_GM.sf2")
        else:
            self.fs.start('alsa')
            sfid = self.fs.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")

        self.fs.program_select(0, sfid, 0, 0)
        self.synthID = self.seq.register_fluidsynth(self.fs)

        self.undo_history = []
        self.redo_history = []

        # Setting NoTransition breaks the "history" screen! Possibly related
        # to some inexplicable rendering bugs on my particular system
        self.manager = ScreenManager(transition=SlideTransition(
                                     duration=.15))

        self.recognizer = Recognizer([])

        # Setup the GestureSurface and bindings to our Recognizer
        surface_screen = Screen(name='surface')

        surface_container = NotePadContainer()
        surface = surface_container.ids['surface']

        surface.bind(on_gesture_discard=self.handle_gesture_discard)
        surface.bind(on_gesture_complete=self.handle_gesture_complete)
        surface.bind(on_gesture_cleanup=self.handle_gesture_cleanup)
        self.surface = surface

        surface_screen.add_widget(surface_container)
        self.manager.add_widget(surface_screen)

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
        layout.add_widget(NotePadMenu())
        layout.add_widget(self.manager)
        layout.add_widget(MainMenu())
        return layout


if __name__ in ('__main__', '__android__'):
    MultistrokeApp().run()
