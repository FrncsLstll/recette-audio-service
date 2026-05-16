import os
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


@app.route('/health')
def health():
    return 'ok'


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if AUTH_TOKEN and request.headers.get('X-Auth-Token') != AUTH_TOKEN:
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
