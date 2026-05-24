import os
import re
import base64
import subprocess
import tempfile
import requests
from flask import Flask, request, jsonify
import yt_dlp

app = Flask(__name__)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
AUTH_TOKEN = os.environ.get('AUTH_TOKEN', '')

SUPPORTED_MIME = {
    'm4a': 'audio/mp4',
    'webm': 'audio/webm',
    'mp3': 'audio/mpeg',
    'mp4': 'video/mp4',
    'ogg': 'audio/ogg',
    'wav': 'audio/wav',
    'flac': 'audio/flac',
}

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB limite Groq Whisper


def auth_check():
    if AUTH_TOKEN and request.headers.get('X-Auth-Token') != AUTH_TOKEN:
        return False
    return True


@app.route('/health')
def health():
    return 'ok'


@app.route('/subtitles', methods=['POST'])
def get_subtitles():
    """Récupère les sous-titres YouTube sans télécharger l'audio."""
    if not auth_check():
        return jsonify({'error': 'Non autorisé'}), 401

    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL manquante'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            'skip_download': True,
            'writeautomaticsub': True,
            'writesubtitles': True,
            'subtitleslangs': ['fr', 'fr-FR', 'en', 'en-US'],
            'subtitlesformat': 'vtt',
            'outtmpl': os.path.join(tmpdir, 'sub.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 20,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            return jsonify({'error': f'Échec sous-titres: {str(e)}'}), 500

        # Chercher le fichier .vtt téléchargé
        vtt_file = None
        for f in os.listdir(tmpdir):
            if f.endswith('.vtt'):
                vtt_file = os.path.join(tmpdir, f)
                break

        if not vtt_file:
            return jsonify({'error': 'Aucun sous-titre disponible'}), 404

        with open(vtt_file, 'r', encoding='utf-8') as f:
            vtt_content = f.read()

        # Nettoyer le format VTT → texte brut
        lines = vtt_content.split('\n')
        text_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('WEBVTT') or '-->' in line:
                continue
            if re.match(r'^\d+$', line):
                continue
            # Supprimer les balises HTML <c>, <b>, etc.
            line = re.sub(r'<[^>]+>', '', line)
            if line and line not in text_lines[-1:]:
                text_lines.append(line)

        transcript = ' '.join(text_lines).strip()
        return jsonify({'transcript': transcript})


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Télécharge l'audio et le transcrit avec Groq Whisper."""
    if not auth_check():
        return jsonify({'error': 'Non autorisé'}), 401

    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL manquante'}), 400

    if not GROQ_API_KEY:
        return jsonify({'error': 'GROQ_API_KEY non configurée'}), 500

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
            'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            return jsonify({'error': f'Téléchargement impossible: {str(e)}'}), 500

        audio_file = None
        for f in os.listdir(tmpdir):
            audio_file = os.path.join(tmpdir, f)
            break

        if not audio_file or not os.path.exists(audio_file):
            return jsonify({'error': 'Aucun fichier audio téléchargé'}), 500

        if os.path.getsize(audio_file) > MAX_FILE_SIZE:
            return jsonify({'error': 'Vidéo trop longue (fichier > 25 Mo)'}), 500

        ext = os.path.splitext(audio_file)[1].lstrip('.').lower()
        mime = SUPPORTED_MIME.get(ext, 'audio/mpeg')

        try:
            with open(audio_file, 'rb') as f:
                resp = requests.post(
                    'https://api.groq.com/openai/v1/audio/transcriptions',
                    headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
                    files={'file': (f'audio.{ext}', f, mime)},
                    data={'model': 'whisper-large-v3-turbo'},
                    timeout=120,
                )
        except Exception as e:
            return jsonify({'error': f'Erreur réseau Groq: {str(e)}'}), 500

        if not resp.ok:
            return jsonify({'error': f'Transcription échouée: {resp.text}'}), 500

        transcript = resp.json().get('text', '')
        return jsonify({'transcript': transcript})


@app.route('/frames', methods=['POST'])
def extract_frames():
    """Récupère l'URL directe de la vidéo, télécharge seulement ~10 Mo, extrait des frames."""
    if not auth_check():
        return jsonify({'error': 'Non autorisé'}), 401

    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL manquante'}), 400

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return jsonify({'error': 'ffmpeg non disponible'}), 500

    with tempfile.TemporaryDirectory() as tmpdir:
        # Étape 1 : récupérer l'URL directe de la vidéo sans télécharger
        video_url = None
        video_duration = None
        try:
            ydl_opts = {'quiet': True, 'socket_timeout': 15, 'format': 'worstvideo[ext=mp4]/worst[ext=mp4]/worstvideo/worst'}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_duration = info.get('duration')  # durée en secondes
                # Chercher l'URL du format le plus léger avec vidéo
                for fmt in sorted(info.get('formats', []), key=lambda x: x.get('filesize') or x.get('tbr') or 9999):
                    if fmt.get('vcodec') not in (None, 'none') and fmt.get('url'):
                        video_url = fmt['url']
                        break
                if not video_url:
                    video_url = info.get('url')
        except Exception as e:
            return jsonify({'error': f'Info vidéo impossible: {str(e)}'}), 500

        if not video_url:
            return jsonify({'error': 'URL vidéo non trouvée'}), 500

        # Étape 2 : télécharger seulement les 12 premiers Mo (couvre ~20-30s à basse qualité)
        video_path = os.path.join(tmpdir, 'video.mp4')
        try:
            resp = requests.get(
                video_url,
                headers={'Range': 'bytes=0-12582912'},  # 12 MB
                timeout=25,
                stream=True
            )
            with open(video_path, 'wb') as f:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= 12_000_000:
                        break
        except Exception as e:
            return jsonify({'error': f'Téléchargement partiel impossible: {str(e)}'}), 500

        if not os.path.exists(video_path) or os.path.getsize(video_path) < 10_000:
            return jsonify({'error': 'Fichier vidéo trop petit'}), 500

        # Étape 3 : extraction dense dans les DERNIÈRES 8 secondes (toutes les 0.5s)
        if video_duration and video_duration > 10:
            d = float(video_duration)
            # 2 frames de contexte
            context_ts = [max(2.0, d * 0.10), d * 0.45]
            # Dernières 8 secondes : une frame toutes les 0.5s
            last_start = max(d - 8.0, d * 0.60)
            end_ts = []
            t = last_start
            while t < d - 0.1 and len(end_ts) < 16:
                end_ts.append(round(t, 1))
                t += 0.5
            all_ts = context_ts + end_ts
            # Dédupliquer et trier
            seen = set()
            timestamps = []
            for t in sorted(all_ts):
                key = round(t, 1)
                if key not in seen:
                    seen.add(key)
                    timestamps.append(t)
        else:
            timestamps = [2, 8, 15, 25, 35, 42, 45]

        frames = []
        frame_debug = []
        for t in timestamps:
            frame_path = os.path.join(tmpdir, f'frame_{t:.1f}.jpg')
            try:
                subprocess.run(
                    [ffmpeg_exe, '-ss', f'{t:.2f}', '-i', video_path,
                     '-vframes', '1', '-q:v', '2', '-vf', 'scale=720:-1',
                     '-y', frame_path],
                    capture_output=True, timeout=10, check=False
                )
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 1000:
                    with open(frame_path, 'rb') as f:
                        frames.append(base64.b64encode(f.read()).decode('utf-8'))
                    frame_debug.append(t)
            except Exception:
                continue

        if not frames:
            return jsonify({'error': 'Impossible d\'extraire des frames', 'duration': video_duration, 'timestamps_tried': timestamps}), 500

        # Les frames sont dans l'ordre chronologique ; on renvoie aussi les métadonnées
        return jsonify({
            'frames': frames,
            'duration': video_duration,
            'timestamps': frame_debug,
        })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
