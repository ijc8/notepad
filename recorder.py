import pyaudio
import wave
import sys

if len(sys.argv) != 3:
    exit(f"usage: {sys.argv[0]} <duration> <output_file>")

length = int(sys.argv[1])
outfile = sys.argv[2]

audio = pyaudio.PyAudio()
 
sr = 44100
frame_size = 1024
stream = audio.open(format=pyaudio.paInt16, channels=1,
                    rate=sr, input=True,
                    frames_per_buffer=frame_size)

frames = []
for i in range(0, int(sr / frame_size * length)):
    data = stream.read(frame_size)
    frames.append(data)
 
stream.stop_stream()
stream.close()
audio.terminate()

f = wave.open(outfile, 'wb')
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(sr)
f.writeframes(b''.join(frames))
f.close()
