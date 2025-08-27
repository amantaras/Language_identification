import click
from .language_detect import detect_languages
from .transcribe import transcribe_segments


@click.group()
def main():
    """Language switch aware transcription CLI."""


@main.command()
@click.option("--audio", required=True, type=click.Path(exists=True))
@click.option(
    "--languages",
    multiple=True,
    required=True,
    help="List of languages to detect e.g. en-US ar-SA",
)
@click.option(
    "--lid-host", required=True, help="ws:// host:port of language id container"
)
@click.option("--out", "out_segments", required=True, type=click.Path())
@click.option(
    "--timeout-sec",
    type=float,
    default=None,
    help="Optional overall timeout for detection loop.",
)
@click.option(
    "--min-segment-sec",
    type=float,
    default=0.0,
    help="Drop segments shorter than this duration (seconds).",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging to stderr.")
def detect_segments(
    audio, languages, lid_host, out_segments, timeout_sec, min_segment_sec, verbose
):
    """Run language detection and produce a segments JSON."""
    import logging
    import sys

    if verbose:
        logging.basicConfig(
            stream=sys.stderr, level=logging.DEBUG, format="[%(levelname)s] %(message)s"
        )
    detect_languages(
        audio_file=audio,
        lid_host=lid_host,
        languages=list(languages),
        out_segments=out_segments,
        timeout_sec=timeout_sec,
        min_segment_sec=min_segment_sec,
    )
    click.echo(f"Wrote segments to {out_segments}")


@main.command()
@click.option("--audio", required=True, type=click.Path(exists=True))
@click.option("--segments", required=True, type=click.Path(exists=True))
@click.option(
    "--map",
    "lang_map",
    multiple=True,
    help="Mapping language=ws://host:port of STT container",
)
@click.option("--key", required=True, help="Central Speech resource key")
@click.option("--billing", required=True, help="Central Speech Billing endpoint")
@click.option("--out", required=True, type=click.Path())
def transcribe(audio, segments, lang_map, key, billing, out):
    """Transcribe existing segments using per-language container endpoints."""
    mapping = {}
    for m in lang_map:
        if "=" not in m:
            raise click.ClickException(f"Invalid map entry: {m}")
        lang, host = m.split("=", 1)
        mapping[lang] = host
    transcribe_segments(
        audio_file=audio,
        segments_json=segments,
        language_host_map=mapping,
        key=key,
        billing=billing,
        out_path=out,
    )
    click.echo(f"Wrote transcript to {out}")


@main.command()
@click.option("--audio", required=True, type=click.Path(exists=True))
@click.option(
    "--languages",
    multiple=True,
    required=True,
    help="Candidate languages (avoid multiple locales per base language).",
)
@click.option("--lid-host", required=True)
@click.option(
    "--map",
    "lang_map",
    multiple=True,
    help="Mapping language=ws://host:port of STT container",
)
@click.option("--key", required=True)
@click.option("--billing", required=True)
@click.option("--out", required=True, type=click.Path())
@click.option(
    "--timeout-sec",
    type=float,
    default=None,
    help="Optional overall timeout for detection loop.",
)
@click.option(
    "--min-segment-sec",
    type=float,
    default=0.0,
    help="Drop segments shorter than this duration (seconds).",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging to stderr.")
def full(
    audio,
    languages,
    lid_host,
    lang_map,
    key,
    billing,
    out,
    timeout_sec,
    min_segment_sec,
    verbose,
):
    """Run detection then transcription in one step."""
    import tempfile
    import os
    import logging
    import sys

    if verbose:
        logging.basicConfig(
            stream=sys.stderr, level=logging.DEBUG, format="[%(levelname)s] %(message)s"
        )
    with tempfile.TemporaryDirectory() as td:
        seg_path = os.path.join(td, "segments.json")
        detect_languages(
            audio_file=audio,
            lid_host=lid_host,
            languages=list(languages),
            out_segments=seg_path,
            timeout_sec=timeout_sec,
            min_segment_sec=min_segment_sec,
        )
        mapping = {}
        for m in lang_map:
            if "=" not in m:
                raise click.ClickException(f"Invalid map entry: {m}")
            lang, host = m.split("=", 1)
            mapping[lang] = host
        transcribe_segments(
            audio_file=audio,
            segments_json=seg_path,
            language_host_map=mapping,
            key=key,
            billing=billing,
            out_path=out,
        )
    click.echo(f"Wrote transcript to {out}")
