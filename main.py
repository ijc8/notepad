# Built-in modules
import sys
import threading
import itertools

# Kivy
from kivy.core.window import Window
from kivy.app import App
from kivy.uix.button import Button
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
from enum import Enum

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

instruments = {"piano": (0, 0), "guitar": (0, 25), "bass": (0, 35), "drum": (128, 0)}

# TODO: populate this, preferably in a semi-automated way
# also think about what we really want to do re. voicings
chords = {'c': [48, 52, 55],
          'f': [53, 57, 60],
          'g': [43, 47, 50],
          'am': [45, 48, 52]}


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

class Action(Enum):
    NONE = 1
    NOTE = 2

class NotePadApp(App):
    debug = BooleanProperty(False)
    playing = BooleanProperty(False)

    def goto_database_screen(self, *l):
        self.database.import_gdb()
        self.manager.current = "database"

    def handle_gesture_cleanup(self, surface, g, *l):
        self.remove_label(g)

    def handle_gesture_merge(self, surface, g, *l):
        self.remove_label(g)
        self.populate_gesture_action(g.id, None)

    def remove_label(self, g):
        if hasattr(g, "_result_label"):
            self.surface.remove_widget(g._result_label)


    def handle_gesture_discard(self, surface, g, *l):
        # Don't bother creating Label if it's not going to be drawn
        if surface.draw_timeout == 0:
            return

        text = "[b]Discarded:[/b] Not enough input"

        self.remove_label(g)

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
        print("handle_gesture_complete")
        print(len(list(g._strokes.items())))
        dollarResult = self.recognizer.recognize(
            util.convert_to_dollar(g.get_vectors())
        )
        result = util.ResultWrapper(dollarResult)
        result._gesture_obj = g

        self.handle_recognize_complete(result)

    def add_to_history_for_undo_redo_with_group_id(self, group_id, notes):
        self.redo_melody_history = []
        group = list(self.surface.canvas.get_group(group_id))
        gesture_vec = []
        for line in group:
            if isinstance(line, Line):
                gesture_vec.append((group_id, line.points))
        self.undo_melody_history.append((gesture_vec, notes))

    def populate_gesture_action(self, gesture_id, action):
        self.gesture_to_action[gesture_id] = action

        # Update notes
        notes = list(
            map(
                lambda action: action[1],
                filter(
                    lambda action: (action and action[0] == Action.NOTE),
                    self.gesture_to_action.values()
                )
            )
        )

        self.notes = sum(notes, [])
        self.notes.sort(key=lambda note: note.x)

    def handle_recognize_complete(self, result, recomplete=False):
        self.history.add_recognizer_result(result)

        # Don't bother creating Label if it's not going to be drawn
        if self.surface.draw_timeout == 0:
            return

        best = result.best
        g = result._gesture_obj
        action = None
        instrument_prefix = "instrument-"
        chord_prefix = "chord-"

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

        self.remove_label(g)

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
        elif recognized_name.startswith(instrument_prefix):
            g._cleanup_time = -1
            instrument = recognized_name[len(instrument_prefix):]
            points = np.array(sum(g.get_vectors(), []))
            y = points[:,1].mean()
            staff = min((0, 1), key=lambda s: np.abs(self.surface.get_height(s, 4) - y))
            # TODO: add real Staff class instead of doing this ad-hoc stuff.
            self.fs.program_select(staff, self.sfid, *instruments[instrument])
        elif recognized_name.startswith(chord_prefix):
            g._cleanup_time = -1
            chord = recognized_name[len(chord_prefix):]
            points = np.array(sum(g.get_vectors(), []))
            x, y = points.mean(axis=0)
            staff = min((0, 1), key=lambda s: np.abs(self.surface.get_height(s, 4) - y))
            # TODO: add real Staff class instead of doing this ad-hoc stuff.
            self.staff_chords[staff].append((x, chord))
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
            treble_pitches = [64, 65, 67, 69, 71, 72, 74, 76, 77][::-1]
            bass_pitches = [43, 45, 47, 48, 50, 52, 53, 55, 57][::-1]
            pitches_per_staff = [treble_pitches, bass_pitches]
            # TODO: don't do this the dumb way, and move this functionality to NotePadSurface
            lines = range(0, 9)
            staves = [0, 1]
            product = itertools.product(staves, lines)
            staff, line = min(
                product,
                key=lambda p: np.abs(self.surface.get_height(p[0], p[1] / 2) - y),
            )
            pitch = pitches_per_staff[staff][line]
            note = Note(pitch, durations[recognized_name[:-4]], x, staff)
            print(note)

            # Hacky way to change note color to black once it's registered.
            self.set_color_rgba(g.id, BLACK)
            g._cleanup_time = -1
            action = (Action.NOTE, [note])
        elif recognized_name.endswith("rest"):
            points = np.array(sum(g.get_vectors(), []))

            x, y = points.mean(axis=0)
            staves = [0, 1]
            staff = min(
                staves, key=lambda s: np.abs(self.surface.get_height(s, 4.5) - y)
            )
            note = Note(0, durations[recognized_name[:-4]], x, staff)
            print(note)

            # Hacky way to change rest color to black once it's registered.
            self.set_color_rgba(g.id, BLACK)
            g._cleanup_time = -1
            action = (Action.NOTE, [note])

        self.populate_gesture_action(g.id, action)

        self.surface.add_widget(g._result_label)

    def is_unrecognized_gesture(self, name, gesture):
        return name is None or self.too_far_away_from_staff(gesture)

    def too_far_away_from_staff(self, gesture):
        miny = gesture.bbox["miny"]
        maxy = gesture.bbox["maxy"]

        dist = np.inf
        for y in [miny, maxy]:
            for staff_y in [self.surface.height_min, self.surface.height_max]:
                dist = min(dist, abs(y - staff_y))

        return (self.surface.size[1] / 22) < dist

    def start_loop(self):
        self.shouldLoop = True
        self.loopHelper()

    def stop_loop(self):
        self.shouldLoop = False

    def loopHelper(self):
        if not self.shouldLoop:
            return

        def loop_callback(time, event, seq, data):
            if not self.shouldLoop:
                return
            self.loopHelper()

        t = int(self.play())
        callbackID = self.seq.register_client(
            name="loop_callback", callback=loop_callback,
        )

        self.seq.timer(t, dest=callbackID, absolute=False)

    # TODO: Perhaps wrap up this stuff in a Player class.
    def stop(self):
        self.play_pos = 0
        if self.playing:
            self.seq.remove_events()
            self.playing = False

    def pause(self):
        if self.playing:
            self.seq.remove_events()
            self.playing = False
            self.play_pos = self.seq.get_tick() - self.play_start_tick

    def play(self):
        if self.playing:
            return
        self.playing = True
        stave_times = [0, 0]
        self.play_start_tick = self.seq.get_tick()

        for note in self.notes:
            t_duration = self.beats_to_ticks(note.duration)
            time = stave_times[note.staff] - self.play_pos
            if note.pitch > 0 and time >= 0:
                time += self.play_start_tick
                self.seq.note_on(
                    time=int(time),
                    channel=note.staff,
                    key=note.pitch,
                    dest=self.synthID,
                    velocity=127,
                )
                self.seq.note_off(
                    time=int(time + t_duration),
                    channel=note.staff,
                    key=note.pitch,
                    dest=self.synthID,
                )
                # Note that this is the nearest chord *to the left* of the note; the preceding chord holds until the next one replaces it.
                print(self.staff_chords[note.staff])
                preceding_chords = [c for c in self.staff_chords[note.staff] if c[0] < note.x]
                if preceding_chords:
                    preceding_chords.sort(key=lambda c: c[0])
                    active_chord = preceding_chords[-1][1]
                    for pitch in chords[active_chord]:
                        self.seq.note_on(
                            time=int(time),
                            channel=note.staff,
                            key=pitch,
                            dest=self.synthID,
                            velocity=127,
                        )
                        self.seq.note_off(
                            time=int(time + t_duration),
                            channel=note.staff,
                            key=pitch,
                            dest=self.synthID,
                        )
            stave_times[note.staff] += t_duration

        duration = max(stave_times)
        def done_playing(*_):
            self.playing = False
            self.play_pos = 0
        done_playing_callback = self.seq.register_client(name=f"done_playng", callback=done_playing)
        self.seq.timer(time=self.play_start_tick + duration, dest=done_playing_callback)
        return duration

    def save_to_file(self, path):
        gesture_vec = []

        for gesture in self.surface._gestures:
            gesture_vec.append((gesture.id, gesture.get_vectors()))

        with open(path, "wb") as data_file:
            pickle.dump(gesture_vec, data_file)

    def save(self, *l):
        path = self.save_popup.ids.filename.text
        if not path:
            self.save_popup.dismiss()
            self.info_popup.text = "Missing filename"
            self.info_popup.open()
            return

        if not path.lower().endswith(".np"):
            path += ".np"

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

    def clear(self):
        self.undo_melody_history = []
        self.redo_melody_history = []
        self.notes = []
        self.gesture_to_action = {}
        self.surface._gestures = []
        self.surface.canvas.clear()
        self.surface.clear_history()

    def undo(self):
        gesture_id = self.surface.undo()

        if gesture_id:
            populate_gesture_action(gesture_id, None)

    def redo(self):
        self.surface.redo()

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
            self.surface_screen.canvas.after.get_group("recording")[4].rgba = (
                0.9,
                0.2,
                0.2,
                1,
            )
            self.surface_screen.canvas.after.get_group("recording")[5].a = 1

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
            display_data = np.array([np.arange(frame_size) / frame_size, np.frombuffer(data, dtype=np.int16).astype(np.float) / np.iinfo(np.int16).max * np.hanning(frame_size)])
            self.surface_screen.canvas.after.get_group("recording-waveform")[0].points = display_data.T.flat
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
        if not is_desktop:
            return

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

        self.add_to_history_for_undo_redo_with_group_id(group_id, notes)
        self.populate_gesture_action(group_id, notes)

    def transcribe_melody(self, audio, sr):
        if not is_desktop:
            return

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

        self.add_to_history_for_undo_redo_with_group_id(group_id, notes)
        self.populate_gesture_action(group_id, notes)

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
        self.play_start_tick = 0
        self.play_pos = 0
        self.notes = []
        self.staff_chords = [[], []]
        self.gesture_to_action = {}
        self.seq = fluidsynth.Sequencer(time_scale=self.time_scale)
        self.fs = fluidsynth.Synth()
        self.debug = False

        if platform == "linux":
            self.fs.start("alsa")
            self.sfid = self.fs.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")
        elif platform == "macosx":
            self.fs.start("coreaudio")
            self.sfid = self.fs.sfload("/Library/Audio/Sounds/Banks/FluidR3_GM.sf2")
        elif platform == "android":
            self.fs.start("oboe")
            # This assumes you have FluidR3_GM.sf2 in the project directory when running buildozer.
            # (And that the soundfont actually made it into the Android package.)
            self.sfid = self.fs.sfload("FluidR3_GM.sf2")
        else:
            exit("Unsupported platform", platform)

        if self.sfid < 0:
            exit("Couldn't load soundfont.")

        self.fs.program_select(0, self.sfid, 0, 0)
        self.fs.program_select(1, self.sfid, 0, 0)
        self.synthID = self.seq.register_fluidsynth(self.fs)

        self.undo_melody_history = []
        self.redo_melody_history = []
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
        self.surface.bind(on_gesture_merge=self.handle_gesture_merge)

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

        with self.surface.canvas:
            Color(1, 0, 0, 1)
            self.plot = Line(points = [(x/88200 * 100, 100 * np.sin(x / 1000.) + 600) for x in range(0, 88200)])

        return layout


if __name__ in ("__main__", "__android__"):
    NotePadApp().run()
