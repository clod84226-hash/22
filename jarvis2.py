"""
Джарвис: голосовой ассистент для слабого ПК (Windows 10).
Слух: Vosk + WebRTC VAD | Голос: Piper | Мозг: Claude (онлайн) или Ollama (офлайн)
Можно перебивать: начни говорить, пока он отвечает, и он замолчит.
"""
import asyncio
import datetime
import glob
import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
import warnings
import webbrowser

warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop  # noqa: E402  (есть в Python 3.12)

import requests
import sounddevice as sd
import webrtcvad
from vosk import KaldiRecognizer, Model, SetLogLevel

try:  # красивый онлайн-голос (необязательный)
    import edge_tts
    import miniaudio
    HAVE_EDGE = True
except ImportError:
    HAVE_EDGE = False

# ====================== НАСТРОЙКИ ======================
BASE = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL = os.path.join(BASE, "models", "vosk-model-small-ru-0.22")
PIPER_EXE = os.path.join(BASE, "piper", "piper.exe")
# Запасной голос Piper (без интернета): берётся первый .onnx из папки piper
PIPER_VOICE = next(iter(sorted(glob.glob(os.path.join(BASE, "piper", "*.onnx")))), "")
PIPER_SLOW = 1.1      # 1.0 обычная скорость; больше = медленнее и спокойнее

# Основной голос (с интернетом): Дмитрий из edge-tts
EDGE_VOICE = "ru-RU-DmitryNeural"
EDGE_RATE = "-8%"     # медленнее = спокойнее
EDGE_PITCH = "-6Hz"   # ниже = солиднее

ONLINE_MODEL = "claude-haiku-5-5"   # быстрая онлайн-модель
OFFLINE_MODEL = "qwen2.5:1.5b"      # если тормозит, поставь "qwen2.5:0.5b"
OLLAMA_URL = "http://localhost:11434"

# Микрофон: None = подобрать автоматически, или номер устройства из списка
# (python -c "import sounddevice as sd; print(sd.query_devices())")
MIC_DEVICE = None

# True: можно перебивать (нужны НАУШНИКИ).
# False: если слушаешь через колонки, иначе он услышит сам себя.
BARGE_IN = True

SYSTEM = (
    "Ты Джарвис, голосовой ассистент. Отвечай по-русски, коротко: "
    "1-3 предложения. Никакой разметки, списков, эмодзи и звёздочек: "
    "твой ответ будет озвучен голосом. Пиши только русскими буквами, без иероглифов."
)

MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
          "августа", "сентября", "октября", "ноября", "декабря"]
APPS = {"блокнот": "notepad", "калькулятор": "calc", "проводник": "explorer"}
SITES = {"ютуб": "https://youtube.com", "гугл": "https://google.com",
         "браузер": "https://google.com"}


# ====================== ГОЛОС (Piper) ======================
class Speaker:
    """Озвучка потоком. stop() мгновенно обрывает речь и очередь."""

    def __init__(self):
        if not PIPER_VOICE:
            raise SystemExit("Не найден файл голоса .onnx в папке piper")
        with open(PIPER_VOICE + ".json", encoding="utf-8") as f:
            self.rate = json.load(f)["audio"]["sample_rate"]
        self.edge_off_until = 0   # если онлайн-голос сбойнул, ждём до этого времени
        self.gen = 0          # поколение: stop() увеличивает, старое отбрасывается
        self.pending = 0      # сколько фраз ещё не доиграно
        self.lock = threading.Lock()
        self.text_q = queue.Queue()
        self.audio_q = queue.Queue()
        threading.Thread(target=self._synth, daemon=True).start()
        threading.Thread(target=self._play, daemon=True).start()

    @property
    def busy(self):
        return self.pending > 0

    def say(self, text, gen=None):
        text = re.sub(r"[*_#`>]", "", text)
        text = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\s.,!?:;()«»\"'%+\-–—…]", "", text).strip()
        if not re.search(r"[0-9A-Za-zА-Яа-яЁё]", text):
            return
        if not text:
            return
        with self.lock:
            self.pending += 1
        self.text_q.put((self.gen if gen is None else gen, text))

    def stop(self):
        self.gen += 1

    def _done(self):
        with self.lock:
            self.pending -= 1

    def _edge(self, text):
        async def run():
            com = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE, pitch=EDGE_PITCH)
            data = bytearray()
            async for chunk in com.stream():
                if chunk["type"] == "audio":
                    data += chunk["data"]
            return bytes(data)

        mp3 = asyncio.run(asyncio.wait_for(run(), timeout=15))
        pcm = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16,
                               nchannels=1, sample_rate=self.rate)
        return pcm.samples.tobytes()

    def _piper(self, text):
        p = subprocess.run(
            [PIPER_EXE, "-m", PIPER_VOICE, "--length_scale", str(PIPER_SLOW),
             "--output_raw"],
            input=(text + "\n").encode("utf-8"),
            capture_output=True,
            cwd=os.path.dirname(PIPER_EXE),
            creationflags=0x08000000,  # без чёрного окна
        )
        return p.stdout

    def _synth(self):
        while True:
            gen, text = self.text_q.get()
            if gen != self.gen:
                self._done()
                continue
            try:
                audio = None
                if HAVE_EDGE and time.time() > self.edge_off_until:
                    try:
                        audio = self._edge(text)
                    except Exception as e:
                        print("Онлайн-голос недоступен, включаю Piper:", e)
                        self.edge_off_until = time.time() + 30
                if audio is None:
                    audio = self._piper(text)
                self.audio_q.put((gen, audio))
            except Exception as e:
                print("Ошибка голоса:", e)
                self._done()

    def _play(self):
        with sd.RawOutputStream(samplerate=self.rate, channels=1, dtype="int16",
                                latency=0.5) as out:
            while True:
                gen, audio = self.audio_q.get()
                for i in range(0, len(audio), 4096):
                    if gen != self.gen:   # перебили: сбросить буфер звука
                        out.abort()
                        out.start()
                        break
                    out.write(audio[i:i + 4096])
                self._done()


# ====================== МОЗГ ======================
def online_ok():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        socket.create_connection(("api.anthropic.com", 443), timeout=2).close()
        return True
    except OSError:
        return False


def stream_online(messages):
    import anthropic
    client = anthropic.Anthropic(timeout=20)
    with client.messages.stream(model=ONLINE_MODEL, max_tokens=300,
                                system=SYSTEM, messages=messages) as s:
        yield from s.text_stream


def stream_offline(messages):
    r = requests.post(
        OLLAMA_URL + "/api/chat",
        json={"model": OFFLINE_MODEL, "stream": True,
              "messages": [{"role": "system", "content": SYSTEM}] + messages,
              "options": {"num_predict": 150}},
        stream=True, timeout=120)
    for line in r.iter_lines():
        if not line:
            continue
        d = json.loads(line)
        chunk = d.get("message", {}).get("content", "")
        if chunk:
            yield chunk
        if d.get("done"):
            break


BRAIN = {"offline": False}


def ask(messages):
    """Онлайн, если есть сеть. Любая ошибка до первого слова: переход на офлайн."""
    BRAIN["offline"] = False
    if online_ok():
        got = False
        try:
            for c in stream_online(messages):
                got = True
                yield c
            return
        except Exception as e:
            print("Онлайн не сработал:", e)
            if got:
                return
    BRAIN["offline"] = True
    print("[офлайн-режим]")
    yield from stream_offline(messages)


# ====================== ЛОКАЛЬНЫЕ КОМАНДЫ (мгновенно, без ИИ) ======================
def local_command(low):
    if "который час" in low or "сколько времени" in low:
        n = datetime.datetime.now()
        return f"Сейчас {n.hour} часов {n.minute} минут."
    if "какое сегодня число" in low or "какая сегодня дата" in low:
        n = datetime.datetime.now()
        return f"Сегодня {n.day} {MONTHS[n.month - 1]} {n.year} года."
    if "открой" in low or "запусти" in low:
        for name, cmd in APPS.items():
            if name in low:
                subprocess.Popen(cmd, shell=True)
                return f"Открываю {name}."
        for name, url in SITES.items():
            if name in low:
                webbrowser.open(url)
                return f"Открываю {name}."
    if "запомни" in low:
        note = low.split("запомни", 1)[1].strip()
        if note:
            with open(os.path.join(BASE, "notes.txt"), "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {note}\n")
            return "Записал."
    return None


# ====================== МИКРОФОН ======================
def _start_mic(dev, rate, ch, mic_q):
    """Открывает микрофон и отдаёт кадры по 30 мс (16 кГц, моно) в очередь."""
    buf = bytearray()
    state = [None]

    def cb(indata, frames, t, status):
        data = bytes(indata)
        if ch == 2:
            data = audioop.tomono(data, 2, 0.5, 0.5)
        if rate != 16000:
            data, state[0] = audioop.ratecv(data, 2, 1, rate, 16000, state[0])
        buf.extend(data)
        while len(buf) >= 960:            # 480 отсчётов = 30 мс
            mic_q.put(bytes(buf[:960]))
            del buf[:960]

    stream = sd.RawInputStream(device=dev, samplerate=rate, dtype="int16",
                               channels=ch, callback=cb)
    stream.start()
    stream.stop()                         # проверили, что открывается
    return stream


def open_mic(mic_q):
    """Перебирает устройства, частоты и каналы, пока не найдёт рабочий вариант."""
    devs = sd.query_devices()
    apis = sd.query_hostapis()
    cands = []
    if MIC_DEVICE is not None:
        cands.append(MIC_DEVICE)
    else:
        try:
            cands.append(sd.query_devices(kind="input")["index"])
        except Exception:
            pass
        for i, d in enumerate(devs):
            api = apis[d["hostapi"]]["name"]
            if d["max_input_channels"] > 0 and "WDM" not in api and i not in cands:
                cands.append(i)
    last = None
    for dev in cands:
        info = devs[dev]
        api = apis[info["hostapi"]]["name"]
        rates = list(dict.fromkeys([16000, int(info["default_samplerate"]), 48000, 44100]))
        for rate in rates:
            for ch in (1, 2):
                if ch > info["max_input_channels"]:
                    continue
                try:
                    stream = _start_mic(dev, rate, ch, mic_q)
                    print(f"Микрофон: {info['name']} [{api}], {rate} Гц, каналов: {ch}")
                    return stream
                except Exception as e:
                    last = e
    raise SystemExit(f"Не удалось открыть микрофон. Последняя ошибка: {last}")


# ====================== ОСНОВНАЯ ЛОГИКА ======================
speaker = None
history = []
worker = None


def respond(my_gen):
    buf, said, held = "", [], []

    def out(sentence):
        said.append(sentence)
        if BRAIN["offline"]:
            held.append(sentence)   # слабый ноутбук: пока думает, молчим
        else:
            speaker.say(sentence, my_gen)

    try:
        for chunk in ask(list(history)):
            if speaker.gen != my_gen:     # перебили, прекращаем
                break
            buf += chunk
            parts = re.split(r"(?<=[.!?…])\s+", buf)
            buf = parts.pop()
            for sentence in parts:
                out(sentence)
        else:
            if buf.strip():
                out(buf)
            if held and speaker.gen == my_gen:
                for sentence in held:     # теперь процессор свободен: говорим
                    speaker.say(sentence, my_gen)
    except Exception as e:
        print("Ошибка мозга:", e)
        speaker.say("Не могу связаться ни с интернетом, ни с локальной моделью.", my_gen)
    if said:
        history.append({"role": "assistant", "content": " ".join(said)})


def handle(text):
    global worker
    print("Ты:", text)
    speaker.stop()
    if worker and worker.is_alive():
        worker.join(timeout=5)

    low = text.lower()
    if len(low.split()) <= 2 and any(w in low for w in ("стоп", "замолчи", "хватит", "тихо")):
        return
    if any(w in low for w in ("выключись", "до свидания", "отбой")):
        speaker.say("До связи.")
        while speaker.busy:
            time.sleep(0.1)
        os._exit(0)

    reply = local_command(low)
    if reply:
        speaker.say(reply)
        return

    history.append({"role": "user", "content": text})
    while len(history) > 12:
        history.pop(0)
    while history and history[0]["role"] != "user":
        history.pop(0)
    worker = threading.Thread(target=respond, args=(speaker.gen,), daemon=True)
    worker.start()


def main():
    global speaker
    SetLogLevel(-1)
    print("Загружаю модели...")
    speaker = Speaker()
    rec = KaldiRecognizer(Model(VOSK_MODEL), 16000)
    vad = webrtcvad.Vad(3)
    mic_q = queue.Queue()
    mic = open_mic(mic_q)

    speaker.say("Джарвис на связи.")
    print("Слушаю. Ctrl+C для выхода.")
    voiced = 0
    with mic:
        while True:
            data = mic_q.get()
            busy = speaker.busy or (worker is not None and worker.is_alive())
            if busy:
                if not BARGE_IN:
                    continue
                voiced = voiced + 1 if vad.is_speech(data, 16000) else 0
                if voiced >= 8:           # ~240 мс речи = перебивание
                    speaker.stop()
                    voiced = 0
            else:
                voiced = 0
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "").strip()
                if len(text) > 1:
                    handle(text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
