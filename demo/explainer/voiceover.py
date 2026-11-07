"""Narrate the explainer with a neural TTS voice and keep it in sync with the animation.

    python demo/explainer/voiceover.py        # -> demo/explainer/how_it_works_voiced.mp4

Steps:
  1. synthesize one clip per narration line (cached in audio/; re-made when the text or voice changes)
  2. schedule each clip at its on-screen cue; where a line needs longer than its scene allows,
     insert a pause into the animation at that point (written to timing.js, read by explainer.html)
  3. re-render the video with those pauses (render.py), then mix the clips onto it

Needs: pip install edge-tts imageio-ffmpeg playwright (edge-tts uses Microsoft's online neural voices).
"""
import argparse
import asyncio
import json
import subprocess
import wave
from pathlib import Path

import edge_tts
import imageio_ffmpeg

import render

HERE = Path(__file__).resolve().parent
AUDIO = HERE / "audio"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# (id, cue: animation second the line starts at, limit: second by which the line must be finished
#  before the next visual change, text). Seconds are in the un-paused animation timeline.
SEGMENTS = [
    ("s1", 0.5, 5.1, "This is a hybrid AI moderation system. Rules, machine learning, and an LLM, only when it's needed."),
    ("s2", 6.0, 14.5, "The easy answer is to send every comment to an LLM. At ten thousand a day, that's fourteen hours of compute, and a growing bill."),
    ("s3a", 15.3, 16.7, "Instead, each comment stops at the cheapest layer that can decide it."),
    ("s3b", 16.9, 21.1, "First, rules. This insult is disguised with dots, but a rule still catches it, in zero milliseconds."),
    ("s3c", 22.0, 27.9, "Next, a logistic regression model. The score is far past the threshold, so it's blocked in two milliseconds."),
    ("s3d", 28.8, 43.3, "This one sounds polite. Its score sits right next to the threshold, so the model isn't sure, and sends it to the LLM. The LLM spots the insult, flags it, and explains why."),
    ("s4a", 44.6, 53.2, "How does it know it's unsure? Scores close to the threshold go to the LLM. I set a budget, about twenty percent, and training turns that into the width of this band."),
    ("s4b", 53.7, 57.5, "For self-harm, even a faint signal gets a second opinion."),
    ("s5", 58.6, 65.9, "On a day of support tickets, rules handle nine percent, the model seventy, and only one in five reaches the LLM."),
    ("s6a", 67.0, 69.9, "I benchmarked ML only, LLM only, and the hybrid, on three use cases. The hybrid always matches or beats ML alone."),
    ("s6b", 70.0, 79.5, "And it costs about seventy-eight percent less than sending everything to the LLM, answering in two milliseconds."),
    ("s7", 80.4, 86.0, "Use the cheapest layer that can decide confidently. Spend on the LLM only where it adds value."),
]
GAP = 0.3       # silence between consecutive lines
BREATH = 0.4    # extra hold after a line before the scene moves on


def wav_seconds(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


async def synth(text, voice, rate, out_wav):
    mp3 = out_wav.with_suffix(".mp3")
    await edge_tts.Communicate(text, voice, rate=rate).save(str(mp3))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(mp3), "-ar", "44100", "-ac", "1", str(out_wav)], check=True)


def synthesize(voice, rate):
    for sid, _, _, text in SEGMENTS:
        wav, key = AUDIO / f"{sid}.wav", AUDIO / f"{sid}.key"
        stamp = f"{voice}|{rate}|{text}"
        if wav.exists() and key.exists() and key.read_text(encoding="utf-8") == stamp:
            continue
        asyncio.run(synth(text, voice, rate, wav))
        key.write_text(stamp, encoding="utf-8")


def schedule():
    inserts = []  # [original second, extra seconds]
    T = lambda t: t + sum(ex for at, ex in inserts if at <= t)
    plan, prev_end = [], 0.0
    print(f"{'line':5} {'start':>6} {'end':>6} {'pause added':>12}")
    for sid, cue, limit, _ in SEGMENTS:
        dur = wav_seconds(AUDIO / f"{sid}.wav")
        start = max(T(cue), prev_end + GAP)
        end = start + dur
        need = end + BREATH - T(limit)
        if need > 0:
            inserts.append([limit, round(need, 2)])
        plan.append((sid, start))
        prev_end = end
        print(f"{sid:5} {start:6.1f} {end:6.1f} {max(need, 0):11.1f}s")
    return plan, inserts


def media_seconds(path):
    info = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True, errors="replace").stderr
    h, m, s = info.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def mix(video, plan, out):
    total = media_seconds(video)
    inputs, filters = [], []
    for i, (sid, start) in enumerate(plan):
        inputs += ["-i", str(AUDIO / f"{sid}.wav")]
        ms = int(start * 1000)
        filters.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
    mix_in = "".join(f"[a{i}]" for i in range(len(plan)))
    filters.append(f"{mix_in}amix=inputs={len(plan)}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,"
                   f"aresample=48000,apad=whole_dur={total:.3f},atrim=0:{total:.3f}[aout]")
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(video), *inputs,
         "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
         "-movflags", "+faststart", str(out)],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-US-AndrewNeural")
    ap.add_argument("--rate", default="+4%")
    ap.add_argument("--out", default=str(HERE / "how_it_works_voiced.mp4"))
    ap.add_argument("--mix-only", action="store_true", help="reuse the rendered how_it_works.mp4")
    a = ap.parse_args()
    AUDIO.mkdir(exist_ok=True)

    synthesize(a.voice, a.rate)
    plan, inserts = schedule()
    (HERE / "timing.js").write_text(f"window.TIMING_INSERTS = {json.dumps(inserts)};\n", encoding="utf-8")
    print(f"pauses inserted: {inserts}")

    silent = HERE / "how_it_works.mp4"
    if not a.mix_only:
        render.video(30, silent)
    mix(silent, plan, Path(a.out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
