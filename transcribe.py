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
    # bt = es.BeatTrackerMultiFeature(minTempo=bpm - 10, maxTempo=bpm + 10)
    # ticks, confidence = bt(audio)
    # For the moment, we'll assume the first beat is right at the start of the audio and we'll follow the tempo exactly.
    num_beats = int(np.ceil(len(audio) / sr * (bpm / 60))) + 1
    print(f'Generating {num_beats} beats.')
    ticks = np.arange(num_beats) / (bpm / 60)
    confidence = None

    # Now put times in terms of beats.
    events = onsets * (bpm / 60)
    beats = ticks * (bpm / 60)
    if verbose:
        print(events)
        print(beats)
    # Quantize event onsets using beats.
    event_index = 0
    beat_index = 0
    out = []
    while event_index < len(events):
        if verbose:
            print(event_index, beat_index, out)
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

    return out


def freq_to_pitch(freq):
    freq = np.ascontiguousarray(freq).copy()
    valid = freq > 0
    freq[valid] = np.log2(freq[valid] / 440) * 12 + 69
    return freq


def extract_melody(audio, sr, bpm, quantization_unit=0.5, verbose=False):
    hop_size = 128

    frame_size = int(sr * quantization_unit / (bpm / 60)) // 4

    freqs, confidence = es.PitchMelodia(hopSize=hop_size, sampleRate=sr, frameSize=frame_size)(audio)
    pitches = freq_to_pitch(freqs)
    if verbose:
        fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, sharex=True)
        time = np.arange(len(audio)) / sr
        ax0.set_title('Input')
        ax0.plot(time, audio)
        time = np.arange(len(pitches)) / (sr / hop_size)
        ax1.set_title('Pitches')
        ax1.plot(time, pitches)
        ax2.set_title('Confidence')
        ax2.plot(time, confidence)
        ax3.set_title('Pitch contour segmentation')
        ax4.set_title('Quantized melody')

    min_duration = quantization_unit / (bpm / 60)
    notes = np.array(es.PitchContourSegmentation(hopSize=hop_size, minDuration=min_duration, sampleRate=sr, rmsThreshold=-2)(freqs, audio)).T
    print(notes)

    if verbose:
        reconstructed = np.zeros(freqs.shape)
        for start, duration, pitch in notes:
            factor = sr / hop_size
            reconstructed[int(start * factor):int((start + duration) * factor)] = pitch
        ax3.plot(time, reconstructed)

    quantized_notes = []
    prev_end = 0
    for start, duration, pitch in notes:
        if verbose:
            print(start, duration, pitch)
        start = quantize(start * (bpm / 60), quantization_unit)
        value = quantize(duration * (bpm / 60), quantization_unit)
        if verbose:
            print(start, value, pitch)
        # Note that this explicitly includes rests as notes with pitch = 0.
        if start != prev_end:
            quantized_notes.append((prev_end, start - prev_end, 0))
        prev_end = start + value
        quantized_notes.append((start, value, pitch))

    if verbose:
        reconstructed = np.zeros(freqs.shape)
        for start, duration, pitch in quantized_notes:
            factor = (60 / bpm) * (sr / hop_size)
            reconstructed[int(start * factor):int((start + duration) * factor)] = pitch
        ax4.plot(time, reconstructed)
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