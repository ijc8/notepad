import essentia
import essentia.standard as es
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import itertools


def quantize(x, quantization_unit):
    return round(x / quantization_unit) * quantization_unit


def extract_rhythm(audio, sr, bpm, quantization_unit=0.5, verbose=False):
    frame_size = 1024
    hop_size = 512

    # Detect onsets.
    od = es.OnsetDetection(method='hfc')
    w = es.Windowing(type='hann')
    fft = es.FFT()
    c2p = es.CartesianToPolar()
    onset_curve = []
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
        mag, phase, = c2p(fft(w(frame)))
        onset_curve.append(od(mag, phase))
    onset_curve = np.array(onset_curve)

    onsets = es.Onsets()(np.array([onset_curve]), [1])

    # Determine location of beats.
    bt = es.BeatTrackerMultiFeature()
    ticks, confidence = bt(audio)

    # Now put times in terms of beats.
    events = onsets * (bpm / 60)
    beats = ticks * (bpm / 60)
    # Quantize event onsets using beats.
    event_index = 0
    beat_index = 0
    out = []
    while event_index < len(events):
        prev_beat = beats[beat_index]
        next_beat = beats[beat_index+1]
        event = events[event_index]
        if event < prev_beat:
            event_index += 1
        elif event > next_beat:
            beat_index += 1
        else:
            offset = (event - prev_beat) / (next_beat - prev_beat)
            quant_offset = quantize(offset, quantization_unit)
            out.append(beat_index + quant_offset)
            event_index += 1

    out = np.array(out)

    if verbose:
        time = np.arange(len(audio)) / sr
        fig, (wf, ax0, ax1, ax2) = plt.subplots(4, sharex=True)
        wf.set_title('Audio')
        wf.plot(time, audio)
        ax0.set_title('Onsets')
        ax0.plot(time, np.repeat(onset_curve / np.max(onset_curve), hop_size)[:len(time)])
        ax0.vlines(onsets, 0, 2)
        ax1.set_title('Quantized Events')
        ax1.vlines((out + beats[0]) / (bpm / 60), 0, 1)
        ax2.set_title(f'Beats (confidence = {confidence})')
        ax2.vlines(ticks, 0, 1)
        plt.show()

    # For the moment, we're calling the first event beat 0.
    return out - out[0]


def extract_melody(audio, sr, bpm, quantization_unit=0.5, verbose=False):
    hop_size = 128

    pitches, confidence = es.PitchMelodia(hopSize=hop_size)(audio)

    if verbose:
        time = np.arange(len(pitches)) / (sr / hop_size)
        fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True)
        ax0.set_title('Pitches (with and without smoothing)')
        ax0.plot(time, pitches)
        ax1.set_title('Confidence')
        ax1.plot(time, confidence)

    filter_size = int(sr / hop_size * (60 / bpm) * quantization_unit)
    if filter_size % 2 == 0:
        filter_size -= 1
    pitches = scipy.signal.medfilt(pitches, filter_size)

    notes = [(k, len(list(v)) / (sr / hop_size) * (bpm / 60)) for k, v in itertools.groupby(pitches)]
    quantized_notes = []
    # Note that this is relative to the start of the first apparent pitch.
    # It is deliberately unquantized, so that we don't have missing time due to discarded pitches.
    t = 0
    for freq, duration in notes:
        if duration > quantization_unit / 2:
            # It's long enough to consider.
            start = quantize(t, quantization_unit)
            value = quantize(duration, quantization_unit)
            # Note that this explicitly includes rests as notes with pitch = 0.
            pitch = int(round(np.log2(freq / 440) * 12 + 69)) if freq > 0 else 0
            quantized_notes.append((start, value, pitch))
        t += duration

    if verbose:
        ax0.plot(time, pitches)
        reconstructed = np.zeros(pitches.shape)
        for start, duration, pitch in quantized_notes:
            factor = (60 / bpm) * (sr / hop_size)
            freq = 2**((pitch - 69)/12) * 440 if pitch > 0 else 0
            reconstructed[int(start * factor):int((start + duration) * factor)] = freq
        ax2.set_title('Melody after quantization')
        ax2.plot(time, reconstructed)
        plt.show()

    return quantized_notes


def test_extract_rhythm():
    audio = es.MonoLoader(filename='test/clap_rhythm.wav', sampleRate=44100)()
    rhythm = extract_rhythm(audio, 44100, 120)
    assert((rhythm == [0, 4, 5.5, 6, 8, 9.5, 10, 11, 12]).all())


def test_extract_melody():
    audio = es.MonoLoader(filename='test/midi_melody.wav', sampleRate=44100)()
    melody = extract_melody(audio, 44100, 120)
    # TODO more testing
    assert((melody == [(0.0, 1.0, 64), (1.0, 1.0, 67), (2.0, 1.0, 69), (3.0, 0.5, 0), (3.5, 1.0, 64), (4.5, 1.0, 67), (5.5, 0.5, 70), (6.0, 0.5, 69), (7.0, 1.0, 0), (8.0, 1.0, 64), (9.0, 1.0, 67), (10.0, 1.0, 69), (11.0, 0.5, 0), (11.5, 1.0, 67), (12.5, 0.5, 64), (13.5, 1.5, 0)]))


if __name__ == '__main__':
    test_extract_rhythm()
    test_extract_melody()