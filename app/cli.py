import json
import click
from typing import List
from .language_detect import detect_languages
from .transcribe import transcribe_segments

@click.group()
def main():
    """Language switch aware transcription CLI."""

@main.command()
@click.option('--audio', required=True, type=click.Path(exists=True))
@click.option('--languages', multiple=True, required=True, help='List of languages to detect e.g. en-US ar-SA')
@click.option('--lid-host', required=True, help='ws:// host:port of language id container')
@click.option('--out', 'out_segments', required=True, type=click.Path())
def detect_segments(audio, languages, lid_host, out_segments):
    """Run language detection and produce a segments JSON."""
    detect_languages(audio_file=audio, lid_host=lid_host, languages=list(languages), out_segments=out_segments)
    click.echo(f"Wrote segments to {out_segments}")

@main.command()
@click.option('--audio', required=True, type=click.Path(exists=True))
@click.option('--segments', required=True, type=click.Path(exists=True))
@click.option('--map', 'lang_map', multiple=True, help='Mapping language=ws://host:port of STT container')
@click.option('--key', required=True, help='Central Speech resource key')
@click.option('--billing', required=True, help='Central Speech Billing endpoint')
@click.option('--out', required=True, type=click.Path())
def transcribe(audio, segments, lang_map, key, billing, out):
    """Transcribe existing segments using per-language container endpoints."""
    mapping = {}
    for m in lang_map:
        if '=' not in m:
            raise click.ClickException(f"Invalid map entry: {m}")
        lang, host = m.split('=', 1)
        mapping[lang] = host
    transcribe_segments(audio_file=audio, segments_json=segments, language_host_map=mapping, key=key, billing=billing, out_path=out)
    click.echo(f"Wrote transcript to {out}")

@main.command()
@click.option('--audio', required=True, type=click.Path(exists=True))
@click.option('--languages', multiple=True, required=True)
@click.option('--lid-host', required=True)
@click.option('--map', 'lang_map', multiple=True, help='Mapping language=ws://host:port of STT container')
@click.option('--key', required=True)
@click.option('--billing', required=True)
@click.option('--out', required=True, type=click.Path())
def full(audio, languages, lid_host, lang_map, key, billing, out):
    """Run detection then transcription in one step."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        seg_path = os.path.join(td, 'segments.json')
        detect_languages(audio_file=audio, lid_host=lid_host, languages=list(languages), out_segments=seg_path)
        mapping = {}
        for m in lang_map:
            if '=' not in m:
                raise click.ClickException(f"Invalid map entry: {m}")
            lang, host = m.split('=', 1)
            mapping[lang] = host
        transcribe_segments(audio_file=audio, segments_json=seg_path, language_host_map=mapping, key=key, billing=billing, out_path=out)
    click.echo(f"Wrote transcript to {out}")
