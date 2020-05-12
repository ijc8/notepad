# Built-in modules
import sys
import threading
import itertools

# Kivy
from kivy.core.window import Window
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from gesturesurface import StrokeSurface
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
# import cProfile


# TODO: mobile support for recording, transcription
is_desktop = platform in ("windows", "macosx", "linux")
if is_desktop:
    import pyaudio

from dollarpy import Recognizer

# Local libraries
from historymanager import GestureHistoryManager
from gesturedatabase import GestureDatabase
from settings import MultistrokeSettingsContainer
import util
import common

if is_desktop:
    import transcribe


# These are in terms of number of beats.
durations = {"eighth": 1 / 2, "quarter": 1, "half": 2, "whole": 4}
reverse_durations = {value: key for key, value in durations.items()}

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

class NotePadExportPopup(Popup):
    pass

# Notes:
# - NotePadSurface represents the "surface" state - all of the ink, everything that is externally visible.
#   We need to easily get all of the state to interpret and save.
#   This should handle things like drawing, erasing, undo, and redo, all as "surface-level" operations/
#   The corresponding changes to internal state are attained by reinterpreting the NotePadSurface.
# - NotePadState represents the internal state - staves, notes, chords, instruments.
#   This should be entirely derivable from the NotePadSurface, and should never go out of sync.


class Note:
    def __init__(self, pitch, duration, x):
        self.pitch = pitch
        self.duration = duration
        # Not sure if this should be here or in Staff.
        self.x = x

    def __repr__(self):
        return f"Note({self.pitch}, {self.duration}, {self.x})"


class Staff:
    "Relevant state for a single Staff. Helper for NotePadState."

    CLEF_PITCHES = {
        'treble': [62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79],
        'bass': [41, 43, 45, 47, 48, 50, 52, 53, 55, 57, 59],
        # these are placed to correspond to standard notation
        # bass drum = 35, floor tom = 41, snare = 38, hi mod tom = 48, ride = 51, closed hi hat = 42
        # I filled in the rest with whatever I wanted:
        # cowbell = 56, tambourine = 54, claves = 75, vibraslap = 58, crash = 49
        'rhythm': [56, 54, 35, 75, 41, 58, 38, 49, 48, 51, 42],
    }

    # Default instruments for each clef; can be overriden by specifying the instrument.
    CLEF_INSTRUMENTS = {
        'treble': 'piano',
        'bass': 'piano',
        'rhythm': 'drum',
    }
    def __init__(self, surface, idx, clef):
        self.idx = idx
        # TODO less duplication
        self.line_spacing = surface.line_spacing
        self.lines = [surface.get_height(idx, l/2) for l in range(-1, 10)]
        self.y = surface.get_height(idx, 2)
        self.clef = clef
        self.instrument = None
        self.notes = []
        # List of tuples (x, chord_name).
        self.chords = []

    def get_instrument(self):
        if self.instrument:
            return self.instrument
        return Staff.CLEF_INSTRUMENTS[self.clef]

    def get_next_note_x(self):
        if len(self.notes) == 0:
            # TODO: eliminate magic number
            return 210

        bar_size = 12 * self.line_spacing
        note_padding = (bar_size * self.notes[-1].duration) / 2
        return self.notes[-1].x + note_padding

    def get_closest_line(self, y):
        return min(range(len(self.lines)), key=lambda l: np.abs(self.lines[l] - y))

    def get_closest_pitch(self, y):
        return Staff.CLEF_PITCHES[self.clef][len(self.lines) - 1 - self.get_closest_line(y)]

    def set_clef(self, clef):
        # If the clef changes, fix the pitches of existing notes.
        for note in self.notes:
            if note.pitch:
                pitch_index = Staff.CLEF_PITCHES[self.clef].index(note.pitch)
                note.pitch = Staff.CLEF_PITCHES[clef][pitch_index]
        self.clef = clef

    # Returns list of notes with active chords.
    def get_notes(self):
        chord_idx = 0
        active_chord = None
        out = []
        self.notes.sort(key=lambda note: note.x)
        for note in self.notes:
            # Advance to next chord.
            while chord_idx < len(self.chords) and self.chords[chord_idx][0] < note.x:
                chord_idx += 1
                active_chord = self.chords[chord_idx - 1][1]
            out.append((active_chord, note))
        return out


class NotePadState:
    def __init__(self, surface):
        self.staves = []
        self.staves.append(Staff(surface, 0, 'treble'))
        self.staves.append(Staff(surface, 1, 'bass'))
        self.staves.append(Staff(surface, 2, 'rhythm'))

    def get_closest_staff(self, y):
        return min(self.staves, key=lambda s: np.abs(s.y - y))

    # This will process the recognition of a single area in the canvas.
    # and modify the internal state.
    @staticmethod
    def handle_recognition_result(state, surface, result):
        best = result.best
        g = result._gesture_obj

        # A bit ad-hoc; oh well.
        clef_suffix = "clef"
        instrument_prefix = "instrument-"
        chord_prefix = "chord-"
        note_suffix = "note"
        rest_suffix = "rest"

        name = best["name"]
        if util.is_unrecognized_gesture(best["name"], g, surface):
            # No match or ignored. Leave it onscreen in case it's for the user's benefit.
            util.set_color_rgba(surface, g, BLACK)
            return

        # TODO: more feedback?
        print(name)
        points = np.array(sum(g.strokes, []))
        x, y = points.mean(axis=0)

        # Indicate stroke recognition through color change.
        util.set_color_rgba(surface, g, common.RECOGNITION_COLOR)

        if name == "barline":
            # Currently, we do nothing for this.
            pass
        elif name.endswith(clef_suffix):
            clef = name[:-len(clef_suffix)]
            staff = state.get_closest_staff(y)
            staff.set_clef(clef)
        elif name.startswith(instrument_prefix):
            instrument = name[len(instrument_prefix):]
            staff = state.get_closest_staff(y)
            staff.instrument = instrument
        elif name.startswith(chord_prefix):
            chord = name[len(chord_prefix):]
            staff = state.get_closest_staff(y)
            staff.chords.append((x, chord))
        elif name.endswith(note_suffix):
            # BEGIN: Uncomment to visualize notehead identification.
            # center = points.mean(axis=0)
            points = points[util.reject_outliers(points[:, 1])]
            # new_center = points.mean(axis=0)
            # radius = 5
            # if self.debug:
            #     with self.surface.canvas:
            #         Color(rgba=WHITE)
            #         Ellipse(pos=center - radius, size=(radius * 2, radius * 2))
            #         Color(rgba=BLACK)
            #         Ellipse(pos=new_center - radius, size=(radius * 2, radius * 2))
            # END

            # Compute notehead center, post-outlier rejection.
            x, y = points.mean(axis=0)
            staff = state.get_closest_staff(y)
            pitch = staff.get_closest_pitch(y)
            staff.notes.append(Note(pitch, durations[name[:-len(note_suffix)]], x))
            print(staff.notes[-1])
        elif name.endswith(rest_suffix):
            staff = state.get_closest_staff(y)
            pitch = staff.get_closest_pitch(y)
            staff.notes.append(Note(0, durations[name[:-len(rest_suffix)]], x))
            print(staff.notes[-1])
        elif name == "eighthpair":
            # Special case to support the most common form of beaming.
            # Let's just split it in the middle and process each half as a note.
            left_half = points[points[:, 0] < x]
            right_half = points[points[:, 0] >= x]
            for points in (left_half, right_half):
                points = points[util.reject_outliers(points[:, 1])]
                x, y = points.mean(axis=0)
                staff = state.get_closest_staff(y)
                pitch = staff.get_closest_pitch(y)
                staff.notes.append(Note(pitch, durations["eighth"], x))
                print(staff.notes[-1])


# Manages current interaction mode, maintains "surface-level" undo and redo history.
class NotePadSurface(StrokeSurface):
    def on_kv_post(self, base_widget):
        super().on_kv_post(base_widget)

        self.mode = "write"  # other options are 'erase', 'pan'
        # Draw lines here, because self.size isn't set yet in __init__().
        self.lines = []
        self.staff_spacing = self.size[1] / 12
        self.line_spacing = self.size[1] / 96
        self.total_staff_number = 3

        self.heights = []
        with self.canvas.before:
            Color(rgba=BLACK)
            for staff in range(self.total_staff_number):
                for line in range(5):
                    points = self.get_points(staff, line)
                    self.heights.append(points[1])
                    self.lines.append(Line(points=points))

    def get_height(self, staff_number, line_number):
        return (
            self.size[1]
            - self.staff_spacing * (staff_number + 1)
            - self.line_spacing * (line_number - 2)
        )

    def get_points(self, staff_number, line_number):
        height = self.get_height(staff_number, line_number)
        return [150, height, self.size[0] - 50, height]

    def on_touch_down(self, touch):
        if self.mode == "pan":
            return False  # let ScatterPlane handle it
        return super().on_touch_down(touch, self.mode)

    def on_touch_move(self, touch):
        if self.mode == "pan":
            return False  # let ScatterPlane handle it
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.mode == "pan":
            return False  # let ScatterPlane handle it
        return super().on_touch_up(touch)

    # TODO: move to util?
    # also, perhaps we should get these directly from the gesture database.
    def get_note_gesture(self, value, pitch):
        gesture = self.gdict[reverse_durations[value] + ("note" if pitch else "rest")][0]
        return [[p.x, p.y] for p in gesture]

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

                offset = 20 if pitch > 72 else -20
                next_point = (
                    x_start + 25,
                    self.get_y_from_pitch(staff_number, pitch) + offset,
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

        points = self.get_note_gesture(value, pitch)
        bar_size = 12 * self.line_spacing
        note_padding = (bar_size * (value / 4.0))
        points = util.align_note(points, pitch, value, self.line_spacing)
        new_center_x = x_start + note_padding
        points += (new_center_x, self.get_y_from_pitch(staff_number, pitch))

        self.add_stroke(points)

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


recognition_memo = {}

class StrokeGroup:
    def __init__(self, strokes):
        self.bbox_margin = 20
        self.ids = frozenset(stroke[0] for stroke in strokes)
        self.strokes = [stroke[1] for stroke in strokes]
        all_points = np.array(sum(self.strokes, []))
        self.minx = np.min(all_points[:, 0])
        self.maxx = np.max(all_points[:, 0])
        self.miny = np.min(all_points[:, 1])
        self.maxy = np.max(all_points[:, 1])

    def merge(self, other):
        self.ids = frozenset.union(self.ids, other.ids)
        self.strokes = self.strokes + other.strokes
        # Could merge the bounding boxes more efficiently.
        all_points = np.array(sum(self.strokes, []))
        self.minx = np.min(all_points[:, 0])
        self.maxx = np.max(all_points[:, 0])
        self.miny = np.min(all_points[:, 1])
        self.maxy = np.max(all_points[:, 1])

    def is_intersecting(self, other):
        bb = (self.minx, self.miny, self.maxx, self.maxy)
        return util.is_bbox_intersecting_helper(bb, other.minx, other.miny, self.bbox_margin) or \
               util.is_bbox_intersecting_helper(bb, other.minx, other.maxy, self.bbox_margin) or \
               util.is_bbox_intersecting_helper(bb, other.maxx, other.miny, self.bbox_margin) or \
               util.is_bbox_intersecting_helper(bb, other.maxx, other.maxy, self.bbox_margin)


class NotePadApp(App):
    debug = BooleanProperty(False)
    playing = BooleanProperty(False)

    def goto_database_screen(self, *l):
        self.database.import_gdb()
        self.manager.current = "database"

    def interpret_canvas(self, surface):
        self.state = NotePadState(surface)
        strokes = self.surface.get_strokes()
        groups = [StrokeGroup([stroke]) for stroke in strokes]

        def needs_merging():
            for g in groups:
                for other in groups:
                    if g == other:
                        continue
                    if g.is_intersecting(other):
                        return (g, other)
            return None  # just to be explicit!

        # TODO optimize
        while True:
            val = needs_merging()
            if (val is None):
                break
            (a, b) = val
            a.merge(b)
            groups.remove(b)

        for g in groups:
            result = self.recognition_memo.get(g.ids, None)
            if not result:
                dollarResult = None

                dollarResult1 = self.recognizer.recognize(util.convert_to_dollar(g.strokes))
                dollarResult2 = self.recognizer.recognize(util.convert_to_dollar(g.strokes, flip=True))
                print(dollarResult1, dollarResult2)

                # Take the best one
                # Only notes can be flipped
                if (dollarResult1[1] < dollarResult2[1]) and (dollarResult2[0].endswith('note') or dollarResult2[0] == 'eighthpair'):
                    dollarResult = dollarResult2
                else:
                    dollarResult = dollarResult1

                result = util.ResultWrapper(dollarResult)
                result._gesture_obj = g
                self.recognition_memo[g.ids] = result
                self.history.add_recognizer_result(result)

            NotePadState.handle_recognition_result(self.state, self.surface, result)

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
        self.play_start_tick = self.seq.get_tick()

        total_duration = 0
        for staff in self.state.staves:
            # Configure instrument
            channel = staff.idx
            self.fs.program_select(channel, self.sfid, *instruments[staff.get_instrument()])
            # Play notes in staff
            time = -self.play_pos
            for active_chord, note in staff.get_notes():
                duration = self.beats_to_ticks(note.duration)
                if note.pitch > 0 and time >= 0:
                    # Play the note.
                    play_time = time + self.play_start_tick
                    self.seq.note_on(
                        time=int(play_time),
                        channel=channel,
                        key=note.pitch,
                        dest=self.synthID,
                        velocity=127,
                    )
                    self.seq.note_off(
                        time=int(play_time + duration),
                        channel=channel,
                        key=note.pitch,
                        dest=self.synthID,
                    )
                    # Play the active chord, if any.
                    if active_chord:
                        for pitch in chords[active_chord]:
                            self.seq.note_on(
                                time=int(play_time),
                                channel=channel,
                                key=pitch,
                                dest=self.synthID,
                                velocity=127,
                            )
                            self.seq.note_off(
                                time=int(play_time + duration),
                                channel=channel,
                                key=pitch,
                                dest=self.synthID,
                            )
                time += duration
            total_duration = max(total_duration, time)

        def done_playing(*_):
            self.playing = False
            self.play_pos = 0
        done_playing_callback = self.seq.register_client(name=f"done_playng", callback=done_playing)
        self.seq.timer(time=self.play_start_tick + total_duration, dest=done_playing_callback)
        return total_duration

    def export_to_wav(self, path):
        # Is it a hack? You'd better believe it.
        backups = (self.play_pos, self.seq, self.fs, self.sfid, self.synthID)
        self.seq = fluidsynth.Sequencer(time_scale=self.time_scale, use_system_timer=False)
        self.fs = fluidsynth.Synth()
        self.sfid = self.fs.sfload("FluidR3_GM.sf2")
        self.synthID = self.seq.register_fluidsynth(self.fs)
        sr = 44100
        duration = int(self.play() / self.time_scale * sr)
        chunk_size = int(sr * 60 / self.tempo / 2)
        with wave.open(path, 'wb') as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            for i in range(int(np.ceil(duration / chunk_size))):
                self.seq.process(int(chunk_size / sr * 1000))
                chunk = self.fs.get_samples(chunk_size)
                wav.writeframes(chunk)
        (self.play_pos, self.seq, self.fs, self.sfid, self.synthID) = backups

    def save_to_file(self, path):
        surface = self.surface.serialize()
        with open(path, "wb") as data_file:
            pickle.dump(surface, data_file)

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

    def clear(self):
        self.load_from_file('saved/initial.np')

    def _clear(self):
        self.state = NotePadState(self.surface)
        self.surface.clear()
        self.recognition_memo = {}

    def load(self, filechooser, *l):
        for f in filechooser.selection:
            self.load_from_file(filename=f)
        self.info_popup.text = "Loaded file"
        self.load_popup.dismiss()
        self.info_popup.open()

    def load_from_file(self, filename):
        self._clear()
        self.surface.clear_history()

        serialized_surface = None
        with open(filename, "rb") as data_file:
            serialized_surface = pickle.load(data_file)

        self.surface.populate(serialized_surface)
        self.surface.redraw_all()
        self.interpret_canvas(self.surface)

    def export(self, *l):
        path = self.export_popup.ids.filename.text
        if not path:
            self.export_popup.dismiss()
            self.info_popup.text = "Missing filename"
            self.info_popup.open()
            return

        if not path.lower().endswith(".png") and not path.lower().endswith(".wav"):
            self.export_popup.dismiss()
            self.info_popup.text = "Filename must end with png or wav extension"
            self.info_popup.open()
            return

        if not path.lower().startswith("export/"):
            path = "export/" + path

        if path.lower().endswith(".png"):
            self.surface.export_to_png(path)
        elif path.lower().endswith(".wav"):
            self.export_to_wav(path)

        self.export_popup.dismiss()
        self.info_popup.text = "Exported to a file"
        self.info_popup.open()

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
    def record(self, callback, melodic, save=False):
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
                callback=lambda a, b, c, idx: Clock.schedule_once(lambda dt: self.update_record_signifiers(idx)),
                data=i + 1,
            )
            self.seq.timer(time=time, dest=update_callback, absolute=False)
            if i < 5:
                self.seq.note_on(
                    time=time,
                    absolute=False,
                    channel=9 if melodic else 10,
                    key=60,
                    dest=self.synthID,
                    velocity=80,
                )

        def reset_record_signifiers(*_):
            record_thread.join()
            for color in self.surface_screen.canvas.after.get_group("recording"):
                color.a = 0
            Clock.schedule_once(lambda dt: callback(data, sr))

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

    def generate_group_id(self):
        self.group_id_counter += 1
        return "transcription group {}".format(self.group_id_counter)

    def record_rhythm(self):
        self.record(self.transcribe_rhythm, False)

    def record_melody(self):
        self.record(self.transcribe_melody, True)

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
            melody.append((start, end - start, 72))
        print(melody)
        group_id = self.generate_group_id()
        self.surface.draw_melody(2, self.state.staves[2].get_next_note_x(), melody, group_id)

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
        self.surface.draw_melody(0, self.state.staves[0].get_next_note_x(), melody, group_id)

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

        # Setup for recording prompts:
        self.fs.program_select(9, self.sfid, *instruments["bass"])
        self.fs.program_select(10, self.sfid, *instruments["drum"])
        self.synthID = self.seq.register_fluidsynth(self.fs)

        self.group_id_counter = 0
        self.record_counter = 0

        # Setting NoTransition breaks the "history" screen! Possibly related
        # to some inexplicable rendering bugs on my particular system
        self.manager = ScreenManager(transition=SlideTransition(duration=0.15))

        self.recognizer = Recognizer()

        # Setup the GestureSurface and bindings to our Recognizer
        self.surface_screen = NotePadScreen(name="surface")

        surface_container = self.surface_screen.ids["container"]
        self.surface = surface_container.ids["surface"]

        self.surface.bind(on_canvas_change=self.interpret_canvas)

        self.manager.add_widget(self.surface_screen)

        # History is the list of gestures drawn on the surface
        history = GestureHistoryManager()
        history_screen = Screen(name="history")
        history_screen.add_widget(history)
        self.history = history
        self.manager.add_widget(history_screen)

        # Database is the list of gesture templates in Recognizer
        database = GestureDatabase(recognizer=self.recognizer)
        self.surface.gdict = database.gdict
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
                points = np.array(self.surface.get_note_gesture(value, int(thing == "note")))
                # points = np.array([[p.x, p.y] for p in database.gdict[duration + thing][-1]])
                points = util.align_note(points, (thing == "note"), value, 15)
                entry.ids.gesture.canvas.get_group("gesture")[0].points = list(points.flat)

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
        self.export_popup = NotePadExportPopup()
        self.save_popup.ids.save_btn.bind(on_press=self.save)
        self.load_popup.ids.filechooser.bind(on_submit=self.load)
        self.export_popup.ids.export_btn.bind(on_press=self.export)
        self.info_popup = InformationPopup()

        with self.surface.canvas:
            Color(1, 0, 0, 1)
            self.plot = Line(points = [(x/88200 * 100, 100 * np.sin(x / 1000.) + 600) for x in range(0, 88200)])

        self.state = NotePadState(self.surface)
        self.clear()

        # Show Tutorial as the initial page.
        self.manager.current = 'tutorial'
        return layout


if __name__ in ("__main__", "__android__"):
    # profile = cProfile.Profile()
    # profile.enable()
    NotePadApp().run()
    # profile.disable()
    # profile.dump_stats('notepad.profile')
