# Built-in modules
import sys
import threading
import itertools

# Kivy
from kivy.core.window import Window
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from gesturesurface import GestureSurface
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scatter import ScatterPlane
from kivy.graphics import Ellipse, Color, Line
from kivy.properties import (
    StringProperty,
    BooleanProperty,
    ListProperty,
    ObjectProperty,
)
from kivy.utils import platform

# Other external libraries
import numpy as np
import fluidsynth
import wave
import pickle
import copy

# TODO: mobile support for recording, transcription
is_desktop = platform in ("windows", "macosx", "linux")
if is_desktop:
    import pyaudio

from dollarpy import Recognizer, Template, Point

# Local libraries
from historymanager import GestureHistoryManager
from gesturedatabase import GestureDatabase
from settings import MultistrokeSettingsContainer
import util
import math

if is_desktop:
    import transcribe


# These are in terms of number of beats.
durations = {"eighth": 1 / 2, "quarter": 1, "half": 2, "whole": 4}


WHITE = (1, 1, 1, 1)
BLACK = (0, 0, 0, 1)
RED = (1, 0, 0, 1)


from helpers import InformationPopup


class MainMenu(GridLayout):
    pass


class MultistrokeAppSettings(MultistrokeSettingsContainer):
    pass


class TutorialEntry(BoxLayout):
    name = StringProperty("default")


class Tutorial(BoxLayout):
    pass


class NotePadScreen(Screen):
    pass


class NotePadSavePopup(Popup):
    pass


class NotePadLoadPopup(Popup):
    pass


class IconButton(Button):
    image = StringProperty()
    image_color = ListProperty([0, 0, 0, 1])


class ToggleIconButton(ToggleButton):
    image = StringProperty()
    image_color = ListProperty([0, 0, 0, 1])


class NotePadSurface(GestureSurface):
    def on_kv_post(self, base_widget):
        super().on_kv_post(base_widget)

        self.mode = "write"  # other options are 'erase', 'pan'
        # Draw lines here, because self.size isn't set yet in __init__().
        self.lines = []
        self.staff_spacing = self.size[1] / 12
        self.line_spacing = self.size[1] / 96
        self.total_staff_number = 2

        heights = []
        with self.canvas.before:
            Color(rgba=BLACK)
            for staff in range(self.total_staff_number):
                for line in range(5):
                    points = self.get_points(staff, line)
                    heights.append(points[1])
                    self.lines.append(Line(points=points))

        heights = np.array(heights)
        self.height_max = np.max(heights)
        self.height_min = np.min(heights)

    def get_height(self, staff_number, line_number):
        return (
            self.size[1]
            - self.staff_spacing * (staff_number + 1)
            - self.line_spacing * (line_number - 2)
        )

    def get_points(self, staff_number, line_number):
        height = self.get_height(staff_number, line_number)
        return [0, height, self.size[0], height]

    def on_touch_down(self, touch):
        if self.mode == "pan":
            return False  # let ScatterPlane handle it
        elif self.mode == "erase":
            return True  # for now, eat the event and do nothing
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.mode == "pan":
            return False  # let ScatterPlane handle it
        elif self.mode == "erase":
            return True  # for now, eat the event and do nothing
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.mode == "pan":
            return False  # let ScatterPlane handle it
        elif self.mode == "erase":
            return True  # for now, eat the event and do nothing
        return super().on_touch_up(touch)

    # TODO: move to util?
    # also, perhaps we should get these directly from the gesture database.
    def get_note_gesture(self, value, pitch):
        gesture = []
        for name, duration in durations.items():
            if duration == value:
                filename = name + ("note" if pitch else "rest")
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

    def draw_melody(self, staff_number, x_start, melody, group_id):
        xs = []
        for note in melody:
            xs.append(x_start)
            x_next_start = self.draw_note(staff_number, x_start, note, group_id)
            x_start = x_next_start

        return xs

    def draw_note(self, staff_number, x_start, note, group_id):
        (_, value, pitch) = note
        if value not in durations.values():
            # HACK
            # possiblities: 1.5 -> dotted quarter
            #               2.5 -> ???
            #               3 -> dotted half note
            #               3.5 -> double-dotted half note
            # new plan: 1.5 -> 1 + 1/2
            #           2.5 -> 2 + 1/2
            #             3 -> 2 + 1
            #           3.5 -> 2 + 1 + 1/2
            hack_map = {
                1.5: (1, 1 / 2),
                2.5: (2, 1 / 2),
                3.0: (2, 1),
                3.5: (2, 1, 1 / 2),
            }
            values = hack_map[value]
            last_point = None
            for value in values:
                next_point = (
                    x_start + 25,
                    self.get_y_from_pitch(staff_number, pitch) - 20,
                )
                x_start = self.draw_note(
                    staff_number, x_start, (None, value, pitch), group_id
                )
                if last_point and pitch > 0:
                    with self.canvas:
                        Line(
                            rgba=BLACK,
                            points=last_point + next_point,
                            width=self.line_width,
                            group=group_id,
                        )
                last_point = next_point
            return x_start

        gesture = self.get_note_gesture(value, pitch)
        points = np.array(sum(gesture, []))
        bar_size = 12 * self.line_spacing
        note_padding = (bar_size * (value / 4.0)) / 2
        points = util.align_note(points, pitch, value, self.line_spacing)
        new_center_x = x_start + note_padding
        points += (new_center_x, self.get_y_from_pitch(staff_number, pitch))

        with self.canvas:
            Color(rgba=BLACK)
            Line(points=points.flat, group=group_id, width=self.line_width)

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


class Note:
    def __init__(self, pitch, duration, x, staff):
        self.pitch = pitch
        self.duration = duration
        # TODO: restructure things so this information is higher up (namely at the Staff level).
        self.x = x
        self.staff = staff

    def __repr__(self):
        return f"Note({self.pitch}, {self.duration}, {self.x}, {self.staff})"


class NotepadApp(App):
    debug = BooleanProperty(False)

    def goto_database_screen(self, *l):
        self.database.import_gdb()
        self.manager.current = "database"

    def handle_gesture_cleanup(self, surface, g, *l):
        if hasattr(g, "_result_label"):
            surface.remove_widget(g._result_label)

    def handle_gesture_discard(self, surface, g, *l):
        # Don't bother creating Label if it's not going to be drawn
        if surface.draw_timeout == 0:
            return

        text = "[b]Discarded:[/b] Not enough input"
        g._result_label = Label(
            text=text,
            markup=True,
            size_hint=(None, None),
            center=(g.bbox["minx"], g.bbox["miny"]),
        )
        self.surface.add_widget(g._result_label)

    def set_color_rgba(self, gesture_id, rgba):
        group = list(self.surface.canvas.get_group(gesture_id))
        for i0, i1 in zip(group, group[2:]):
            if isinstance(i0, Color) and isinstance(i1, Line):
                i0.rgba = rgba

    def handle_gesture_complete(self, surface, g, *l):
        dollarResult = self.recognizer.recognize(
            util.convert_to_dollar(g.get_vectors())
        )

        result = util.ResultWrapper(dollarResult)

        result._gesture_obj = g

        self.handle_recognize_complete(result)

    def add_to_history_for_undo_redo_with_group_id(self, group_id, notes):
        self.redo_history = []
        group = list(self.surface.canvas.get_group(group_id))
        gesture_vec = []
        for line in group:
            if isinstance(line, Line):
                gesture_vec.append((group_id, line.points))
        self.undo_history.append((gesture_vec, notes))

    def add_to_history_for_undo_redo(self, gestures, notes):
        self.redo_history = []
        gesture_vec = []
        for gesture in gestures:
            gesture_vec.append((gesture.id, gesture.get_vectors()))
        self.undo_history.append((gesture_vec, notes))

    def handle_recognize_complete(self, result, *l):
        self.history.add_recognizer_result(result)

        # Don't bother creating Label if it's not going to be drawn
        if self.surface.draw_timeout == 0:
            return

        best = result.best
        g = result._gesture_obj

        recognized_name = best["name"]
        if self.is_unrecognized_gesture(best["name"], g):
            # No match or ignored. Leave it onscreen in case it's for the user's benefit.
            self.set_color_rgba(g.id, RED)
            recognized_name = "Not Recognized"
            g._cleanup_time = -1

            # For saving inks
            # self.record_counter += 1
            # filename='ink/record_{}'.format(self.record_counter)
            # with open(filename, "wb") as data_file:
            #    pickle.dump(g.get_vectors(), data_file)

            self.add_to_history_for_undo_redo([g], [])

        text = "[b]%s[/b]" % (recognized_name)

        text = f"[color=#000000]{text}[/color]"
        g._result_label = Label(
            text=text,
            markup=True,
            size_hint=(None, None),
            center=(g.bbox["minx"], g.bbox["miny"]),
        )

        if recognized_name == "trebleclef" or recognized_name == "barline":
            self.set_color_rgba(g.id, BLACK)
            g._cleanup_time = -1
            self.add_to_history_for_undo_redo([g], [])
        elif recognized_name.endswith("note"):
            points = np.array(sum(g.get_vectors(), []))

            # For saving inks
            # with open("ink/eighthnote", "wb") as data_file:
            #    pickle.dump(g.get_vectors(), data_file)

            # TODO: temp visualization
            center = points.mean(axis=0)
            points = points[util.reject_outliers(points[:, 1])]
            new_center = points.mean(axis=0)
            radius = 5
            if self.debug:
                with self.surface.canvas:
                    Color(rgba=WHITE)
                    Ellipse(pos=center - radius, size=(radius * 2, radius * 2))
                    Color(rgba=BLACK)
                    Ellipse(pos=new_center - radius, size=(radius * 2, radius * 2))
            # end tmp

            x, y = points.mean(axis=0)
            pitches = [64, 65, 67, 69, 71, 72, 74, 76, 77][::-1]
            # TODO: don't do this the dumb way, and move this functionality to NotePadSurface
            lines = range(0, 9)
            staves = [0, 1]
            product = itertools.product(staves, lines)
            staff, line = min(
                product,
                key=lambda p: np.abs(self.surface.get_height(p[0], p[1] / 2) - y),
            )
            pitch = pitches[line]
            note = Note(pitch, durations[recognized_name[:-4]], x, staff)
            print(note)

            self.notes.append(note)
            self.notes.sort(key=lambda note: note.x)
            # Hacky way to change note color to black once it's registered.
            self.set_color_rgba(g.id, BLACK)
            g._cleanup_time = -1
            self.add_to_history_for_undo_redo([g], [note])
        elif recognized_name.endswith("rest"):
            points = np.array(sum(g.get_vectors(), []))

            # For saving inks
            # with open("ink/eighthrest", "wb") as data_file:
            #    pickle.dump(g.get_vectors(), data_file)
            x, y = points.mean(axis=0)
            staves = [0, 1]
            staff = min(
                staves, key=lambda s: np.abs(self.surface.get_height(s, 4.5) - y)
            )
            note = Note(0, durations[recognized_name[:-4]], x, staff)
            print(note)
            self.notes.append(note)
            self.notes.sort(key=lambda note: note.x)
            # Hacky way to change rest color to black once it's registered.
            self.set_color_rgba(g.id, BLACK)
            g._cleanup_time = -1
            self.add_to_history_for_undo_redo([g], [note])

        self.surface.add_widget(g._result_label)

    def is_unrecognized_gesture(self, name, gesture):
        if name is None:
            return True

        if self.is_command_gesture(name):
            return False

        if self.too_far_away_from_staff(gesture):
            return True

        return False

    def too_far_away_from_staff(self, gesture):
        miny = gesture.bbox["miny"]
        maxy = gesture.bbox["maxy"]

        dist = np.inf
        for y in [miny, maxy]:
            for staff_y in [self.surface.height_min, self.surface.height_max]:
                dist = min(dist, abs(y - staff_y))

        return (self.surface.size[1] / 22) < dist

    def is_command_gesture(self, name):
        commands = ["play", "loop", "stop"]
        return name in commands

    # TODO: stop playback immediately
    def stop(self):
        self.shouldLoop = False

    def loop(self):
        self.shouldLoop = True
        self.loopHelper()

    def loopHelper(self):
        if not self.shouldLoop:
            return

        def loop_callback(time, event, seq, data):
            if not self.shouldLoop:
                return
            self.loop()

        t = int(self.playback())
        callbackID = self.seq.register_client(
            name="loop_callback", callback=loop_callback,
        )

        self.seq.timer(t, dest=callbackID, absolute=False)

    def playback(self):
        stave_times = [0, 0]
        for note in self.notes:
            t_duration = self.beats_to_ticks(note.duration)
            if note.pitch > 0:
                self.seq.note_on(
                    time=int(stave_times[note.staff]),
                    absolute=False,
                    channel=0,
                    key=note.pitch,
                    dest=self.synthID,
                    velocity=127,
                )
            stave_times[note.staff] += t_duration
        return max(stave_times)

    def save_to_file(self, path):
        gesture_vec = []

        for gesture in self.surface._gestures:
            gesture_vec.append((gesture.id, gesture.get_vectors()))

        with open(path, "wb") as data_file:
            pickle.dump(gesture_vec, data_file)
        return

    def save(self, *l):
        path = self.save_popup.ids.filename.text
        if not path:
            self.save_popup.dismiss()
            self.info_popup.text = "Missing filename"
            self.info_popup.open()
            return

        if not path.lower().endswith(".ntp"):
            path += ".ntp"

        if not path.lower().startswith("saved/"):
            path = "saved/" + path

        self.save_to_file(path)

        self.save_popup.dismiss()
        self.info_popup.text = "Saved to a file"
        self.info_popup.open()
        self.load_popup.ids.filechooser._update_files()

    def load(self, filechooser, *l):
        for f in filechooser.selection:
            self.load_from_file(filename=f)
        self.info_popup.text = "Loaded file"
        self.load_popup.dismiss()
        self.info_popup.open()

    def load_from_file(self, filename):
        self.clear()

        gesture_vec = None
        with open(filename, "rb") as data_file:
            gesture_vec = pickle.load(data_file)

        for val in gesture_vec:
            group_id, vectors = val
            np_vectors = np.array(vectors)
            with self.surface.canvas:
                Color(rgba=BLACK)
                Line(points=np_vectors.flat, group=group_id, width=2)

        return

    def clear(self):
        self.undo_history = []
        self.redo_history = []
        self.notes = []
        self.surface._gestures = []
        self.surface.canvas.clear()

    def undo(self):
        if len(self.undo_history) == 0:
            return
        history_val = self.undo_history[-1]
        self.undo_history.pop()

        gesture_vec, notes = history_val

        for val in gesture_vec:
            group_id, vectors = val
            self.surface.canvas.remove_group(group_id)

        for note in notes:
            if note:
                self.notes.remove(note)

        self.notes.sort(key=lambda note: note.x)
        self.redo_history.append(history_val)

    def redo(self):
        if len(self.redo_history) == 0:
            return

        history_val = self.redo_history[-1]
        self.redo_history.pop()

        gesture_vec, notes = history_val

        for val in gesture_vec:
            group_id, vectors = val
            np_vectors = np.array(vectors)
            with self.surface.canvas:
                Color(rgba=BLACK)
                Line(points=np_vectors.flat, group=group_id, width=2)

        for note in notes:
            self.notes.append(note)
        self.notes.sort(key=lambda note: note.x)

        self.undo_history.append(history_val)

    def update_record_signifiers(self, idx):
        idx -= 1  # fluidsynth scheduler workaround
        if idx < 4:
            self.surface_screen.canvas.after.get_group("recording")[idx].rgba = (
                1,
                0.5,
                0.5,
                0.7,
            )
        else:
            self.surface_screen.canvas.after.get_group("recording")[idx].rgba = (
                0.9,
                0.2,
                0.2,
                1,
            )

    # TODO: we're edging into callback hell here, so maybe it's time to bust out async/await.
    def record(self, callback, save=False):
        if not is_desktop:
            callback(np.zeros(44100 * 2), 44100)
            return
        sr = 44100
        data = np.zeros(int(60 / self.tempo * sr * 4), dtype=np.int)
        record_thread = threading.Thread(
            target=self.record_helper, args=(sr, data, save)
        )
        record_thread.start()

        # Play four beeps to indicate tempo and key.
        for i in range(5):
            self.surface_screen.canvas.after.get_group("recording")[i].rgba = (
                0.8,
                0.7,
                0.7,
                0.3,
            )
            time = self.beats_to_ticks(i + 1)
            # Bizarrely, this cannot accept the value 0 (it's replaced by None).
            update_callback = self.seq.register_client(
                name=f"record_update_callback",
                callback=lambda a, b, c, idx: print("hmm", a, b, c, idx)
                or self.update_record_signifiers(idx),
                data=i + 1,
            )
            self.seq.timer(time=time, dest=update_callback, absolute=False)
            if i < 5:
                self.seq.note_on(
                    time=time,
                    absolute=False,
                    channel=0,
                    key=60,
                    dest=self.synthID,
                    velocity=80,
                )

        def reset_record_signifiers(*_):
            record_thread.join()
            for color in self.surface_screen.canvas.after.get_group("recording"):
                color.a = 0
            callback(data, sr)

        finish_callback = self.seq.register_client(
            name="record_finish_callback", callback=reset_record_signifiers
        )
        self.seq.timer(
            time=self.beats_to_ticks(9), dest=finish_callback, absolute=False
        )

    def record_helper(self, sr, out, save):
        """Helper function. Plays one measure of beats and then records one measure of audio."""
        frame_size = 1024
        # TODO: for now, locked to one measure of recording. figure out actual policy.
        # (10 beats to include the calibration measure + latency allowance on each side)
        length = 10 / (self.tempo / 60)  # seconds
        print("recording")
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sr,
            input=True,
            frames_per_buffer=frame_size,
        )
        print("latencies", stream.get_input_latency(), stream.get_output_latency())
        # for the moment we're assuming pyaudio's output latency is a good estimate of fluidsynth's...
        latency = stream.get_input_latency() + stream.get_output_latency()

        frames = []
        for i in range(0, int(sr / frame_size * length)):
            data = stream.read(frame_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        data = b"".join(frames)
        data = np.frombuffer(data, dtype=np.int16).astype(np.int)
        # Throw out the first five beats plus latency, and the last beat.
        start = int(((60 / self.tempo) * 5 + latency) * sr)
        length = int(60 / self.tempo * sr * 4)
        out[:] = data[start : start + length]

        if save:
            outfile = "recorded.wav"
            f = wave.open(outfile, "wb")
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(data.astype(np.int16).tobytes())
            f.close()
            print(f"saved recording to {outfile}.")

    def calculate_x_start(self):
        if len(self.notes) == 0:
            return 20

        bar_size = 12 * self.surface.line_spacing
        note_padding = (bar_size * self.notes[-1].duration) / 2
        return self.notes[-1].x + note_padding

    def generate_group_id(self):
        self.group_id_counter += 1
        return "transcription group {}".format(self.group_id_counter)

    def record_rhythm(self):
        self.record(self.transcribe_rhythm)

    def record_melody(self):
        self.record(self.transcribe_melody)

    def transcribe_rhythm(self, audio, sr):
        if is_desktop:
            rhythm = list(
                transcribe.extract_rhythm(audio, sr, self.tempo, verbose=self.debug)
            )
            print("rhythm", rhythm)
            # HACK for prototype demo
            melody = []
            for (start, end) in zip(rhythm, rhythm[1:] + [4]):
                melody.append((start, end - start, 64))
            print(melody)
            group_id = self.generate_group_id()
            xs = self.surface.draw_melody(0, self.calculate_x_start(), melody, group_id)
            notes = [
                Note(pitch, value, x, 0) for (_, value, pitch), x in zip(melody, xs)
            ]

            self.notes += notes
            self.add_to_history_for_undo_redo_with_group_id(group_id, notes)

    def transcribe_melody(self, audio, sr):
        if is_desktop:
            melody = transcribe.extract_melody(
                audio, sr, self.tempo, verbose=self.debug
            )
            # For the demo, we're going to keep this in the treble clef: say, in a range of 62 to 79.
            # TODO: draw ledger lines
            def get_in_range(p):
                p = (p - 62) % 24
                if p > 79 - 62:
                    p %= 12
                return p + 62

            melody = [(s, v, get_in_range(p) if p else 0) for s, v, p in melody]
            print(melody)
            group_id = self.generate_group_id()
            xs = self.surface.draw_melody(0, self.calculate_x_start(), melody, group_id)
            notes = [
                Note(pitch, value, x, 0) for (_, value, pitch), x in zip(melody, xs)
            ]
            self.notes += notes
            self.add_to_history_for_undo_redo_with_group_id(group_id, notes)
            print("melody", melody)

    def beats_to_ticks(self, beats):
        ticks = self.time_scale / (self.tempo / 60) * beats
        # TODO: temp for debugging
        assert (ticks == int(ticks), f"{beats} and {ticks}")
        return int(round(ticks))

    def build(self):

        self.time_scale = 1000
        self.tempo = 120  # bpm
        if is_desktop:
            self.audio = pyaudio.PyAudio()
        self.notes = []
        self.seq = fluidsynth.Sequencer(time_scale=self.time_scale)
        self.fs = fluidsynth.Synth()
        self.debug = False

        if platform == "linux":
            self.fs.start("alsa")
            sfid = self.fs.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")
        elif platform == "macosx":
            self.fs.start("coreaudio")
            sfid = self.fs.sfload("/Library/Audio/Sounds/Banks/FluidR3_GM.sf2")
        elif platform == "android":
            self.fs.start("oboe")
            # This assumes you have FluidR3_GM.sf2 in the project directory when running buildozer.
            # (And that the soundfont actually made it into the Android package.)
            sfid = self.fs.sfload("FluidR3_GM.sf2")
        else:
            exit("Unsupported platform", platform)

        if sfid < 0:
            exit("Couldn't load soundfont.")

        self.fs.program_select(0, sfid, 0, 0)
        self.synthID = self.seq.register_fluidsynth(self.fs)

        self.undo_history = []
        self.redo_history = []
        self.group_id_counter = 0
        self.record_counter = 0

        # Setting NoTransition breaks the "history" screen! Possibly related
        # to some inexplicable rendering bugs on my particular system
        self.manager = ScreenManager(transition=SlideTransition(duration=0.15))

        self.recognizer = Recognizer([])

        # Setup the GestureSurface and bindings to our Recognizer
        self.surface_screen = NotePadScreen(name="surface")

        surface_container = self.surface_screen.ids["container"]
        self.surface = surface_container.ids["surface"]

        self.surface.bind(on_gesture_discard=self.handle_gesture_discard)
        self.surface.bind(on_gesture_complete=self.handle_gesture_complete)
        self.surface.bind(on_gesture_cleanup=self.handle_gesture_cleanup)

        self.manager.add_widget(self.surface_screen)

        # History is the list of gestures drawn on the surface
        history = GestureHistoryManager()
        history_screen = Screen(name="history")
        history_screen.add_widget(history)
        self.history = history
        self.manager.add_widget(history_screen)

        # Database is the list of gesture templates in Recognizer
        database = GestureDatabase(recognizer=self.recognizer)
        database_screen = Screen(name="database")
        database_screen.add_widget(database)
        self.database = database
        self.manager.add_widget(database_screen)

        # Settings screen
        app_settings = MultistrokeAppSettings()
        ids = app_settings.ids

        ids.max_strokes.bind(value=self.surface.setter("max_strokes"))
        ids.temporal_win.bind(value=self.surface.setter("temporal_window"))
        ids.timeout.bind(value=self.surface.setter("draw_timeout"))
        ids.line_width.bind(value=self.surface.setter("line_width"))
        ids.draw_bbox.bind(value=self.surface.setter("draw_bbox"))
        ids.use_random_color.bind(value=self.surface.setter("use_random_color"))

        settings_screen = Screen(name="settings")
        settings_screen.add_widget(app_settings)
        self.manager.add_widget(settings_screen)

        tutorial = Tutorial()

        for duration, value in list(durations.items())[::-1]:
            for thing in ("note", "rest"):
                entry = TutorialEntry(name=f"{duration} {thing}")
                tutorial.ids.notegrid.add_widget(entry)
                points = np.array(
                    sum(self.surface.get_note_gesture(value, int(thing == "note")), [])
                )
                # points = np.array([[p.x, p.y] for p in database.gdict[duration + thing][-1]])
                points = util.align_note(points, (thing == "note"), value, 15)
                entry.ids.gesture.canvas.get_group("gesture")[0].points = list(
                    points.flat
                )

        tutorial_screen = Screen(name="tutorial")
        tutorial_screen.add_widget(tutorial)
        self.manager.add_widget(tutorial_screen)

        # Wrap in a gridlayout so the main menu is always visible
        layout = GridLayout(cols=1)
        layout.add_widget(self.manager)
        layout.add_widget(MainMenu())

        def on_keyboard(instance, key, scancode, codepoint, modifiers):
            if codepoint == "d":
                self.debug = not self.debug

        Window.bind(on_keyboard=on_keyboard)

        self.save_popup = NotePadSavePopup()
        self.load_popup = NotePadLoadPopup()
        self.save_popup.ids.save_btn.bind(on_press=self.save)
        self.load_popup.ids.filechooser.bind(on_submit=self.load)
        self.info_popup = InformationPopup()

        return layout


if __name__ in ("__main__", "__android__"):
    NotepadApp().run()
