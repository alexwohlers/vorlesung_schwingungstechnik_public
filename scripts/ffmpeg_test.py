import subprocess
import os

ffmpeg_path = r'\github\_nosync_tools_ffmpeg\bin\ffmpeg.exe'
output_path = 'ffmpeg_test_output.mp4'
cmd = [
    ffmpeg_path,
    '-f', 'lavfi',
    '-i', 'testsrc=duration=2:size=640x480:rate=25',
    '-c:v', 'libx264',
    output_path
]
print(f"Starte ffmpeg: {' '.join(cmd)}")
try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Returncode:", result.returncode)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    if result.returncode == 0:
        print(f"Erfolgreich! Video gespeichert unter: {output_path}")
    else:
        print("Fehler beim ffmpeg-Aufruf!")
except Exception as e:
    print(f"Exception: {e}")
