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


@app.route('/thumbnail', methods=['POST'])
def get_thumbnail():
    """Récupère la miniature d'une vidéo Instagram pour analyse visuelle (sans télécharger la vidéo)."""
    if not auth_check():
        return jsonify({'error': 'Non autorisé'}), 401

    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL manquante'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            'skip_download': True,
            'writethumbnail': True,
            'outtmpl': os.path.join(tmpdir, 'thumb'),
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 20,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            return jsonify({'error': f'Impossible: {str(e)}'}), 500

        # Trouver le fichier miniature téléchargé
        thumb_file = None
        for f in os.listdir(tmpdir):
            fp = os.path.join(tmpdir, f)
            if os.path.isfile(fp):
                thumb_file = fp
                break

        if not thumb_file:
            return jsonify({'error': 'Pas de miniature disponible'}), 404

        with open(thumb_file, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        ext = os.path.splitext(thumb_file)[1].lstrip('.').lower() or 'jpg'
        mime = f'image/{"jpeg" if ext == "jpg" else ext}'
        return jsonify({'thumbnail': b64, 'mime': mime})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
