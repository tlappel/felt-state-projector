"""Local Gradio interface. Run: python -m demo.app --help."""
import argparse
import atexit
import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr

from demo.core import (DemoError, DemoService, ReplayProjector, ROOT,
                       case_choices, packet_for, receiver_request)
from demo.providers import LaptopProjector, NebiusReceiver


def build_app(service):
    choices = case_choices()
    receiver_enabled = service.receiver is not None

    def owner(request):
        if not request.session_hash:
            raise gr.Error('A browser session is required.')
        return request.session_hash

    def reset(case_id, voice, request: gr.Request):
        service.invalidate(owner(request))
        return (packet_for(case_id, voice)['sensorium']['body'], '', '', '',
                'Prepare a whisper for this selection.', {}, False,
                gr.Button(interactive=False), '', {}, {}, 'Not sent.')

    def generate(case_id, voice, request: gr.Request):
        try:
            c = service.generate(owner(request), case_id, voice)
        except DemoError as exc:
            return ('', '', '', str(exc), {}, False, gr.Button(interactive=False), '', {}, {}, 'Not sent.')
        result = c['result']
        status = ('Format and sources pass. Inspect the meaning before sending.' if c['eligible']
                  else 'Delivery blocked: invalid, incomplete, or empty output.')
        if result.get('historical_grade'):
            status += ' Saved review: ' + result['historical_grade'].replace('_', ' ') + '.'
        details = {'mode': c['mode'], 'source_ids': c['source_ids'], 'checks': c['checks'],
                   'ended_with_eos': result.get('ended_with_eos'),
                   'hit_token_cap': result.get('hit_token_cap'),
                   'startup_seconds': result.get('startup_seconds'),
                   'generation_seconds': result.get('demo_generation_seconds'),
                   'saved_evaluation_generation_seconds': result.get('seconds_including_any_compilation')
                   if result.get('historical_grade') else None}
        preview = receiver_request(c['text'], c['source_ids'], c['mode']) if c['eligible'] else {}
        return (c['token'], c['text'], result.get('reply', ''), status, details, False,
                gr.Button(interactive=c['eligible'] and receiver_enabled), '', {}, preview, 'Not sent.')

    def deliver(token, case_id, voice, inspected, request: gr.Request):
        try:
            result = service.deliver(owner(request), token, case_id, voice, inspected)
        except DemoError as exc:
            # A rejected click must not erase a prior response or its receipt.
            # Keep the first failure visible too if this is a subsequent click.
            return gr.skip(), gr.skip(), str(exc)
        return (result['text'], {k: v for k, v in result.items() if k != 'text'},
                'Response received. This candidate has been dispatched; another click will not send it again.')

    with gr.Blocks(title='Felt-State Projector', analytics_enabled=False) as app:
        gr.Markdown('# Felt-State Projector\nA bodily whisper. A character’s own response.')
        gr.Markdown('**Projector:** ' + service.projector.mode)
        gr.Markdown('**Receiver:** ' + (
            f'Live Nebius · at most {service.receiver.limit} attempts in this process.'
            if receiver_enabled else 'Disabled · preview the request without making an API call.'))
        token = gr.State('')
        with gr.Row():
            with gr.Column():
                gr.Markdown('### 1 · Set the body\nChoose among eight previously evaluated synthetic states.')
                case = gr.Dropdown(choices, value=choices[0][1], label='Body state', allow_custom_value=False)
                voice = gr.Radio(['plain', 'conversational'], value='plain', label='Voice')
                body = gr.JSON(packet_for(choices[0][1], 'plain')['sensorium']['body'], label='Supplied body values')
                prepare = gr.Button('Generate whisper' if isinstance(service.projector, LaptopProjector)
                                    else 'Load saved whisper', variant='primary')
            with gr.Column():
                gr.Markdown('### 2 · Inspect the whisper')
                whisper = gr.Textbox(label='Candidate whisper', interactive=False, lines=3)
                status = gr.Textbox(value='Prepare a whisper for this selection.', label='Review status', interactive=False)
                inspected = gr.Checkbox(label='I inspected this whisper and want the character to receive it.', value=False)
                send = gr.Button('Send to character', interactive=False)
        gr.Markdown('### 3 · Let the character respond\n**Fictional scene:** Rowan is sketching at a table on a quiet afternoon, with no deadline. '
                    'The whisper offers context; Rowan chooses a response. This demo cannot perform actions in Numen.')
        response = gr.Textbox(label='Character response', interactive=False, lines=5)
        delivery_status = gr.Textbox(value='Not sent.', label='Delivery status', interactive=False)
        receipt = gr.JSON(label='Receiver receipt — last successful response')
        with gr.Accordion('Inspect inputs and receipts', open=False):
            gr.Markdown('Format checks do not certify meaning. Changing the input clears the candidate. '
                        'Candidates expire after ten minutes. Saved replay never becomes live generation.')
            raw = gr.Textbox(label='Exact projector output', interactive=False, lines=4)
            details = gr.JSON(label='Projector receipt')
            preview = gr.JSON(label='Exact proposed receiver request')
        generated = [token, whisper, raw, status, details, inspected, send, response, receipt, preview, delivery_status]
        prepare.click(generate, [case, voice], generated, api_name='prepare',
                      api_visibility='undocumented', concurrency_id='demo', concurrency_limit=1)
        for component in (case, voice):
            component.input(reset, [case, voice], [body] + generated,
                            api_visibility='private', concurrency_id='demo', concurrency_limit=1)
        send.click(deliver, [token, case, voice, inspected], [response, receipt, delivery_status],
                   api_name='deliver', api_visibility='undocumented', concurrency_id='demo', concurrency_limit=1)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projector', choices=['replay', 'laptop'], default='replay')
    parser.add_argument('--training-python', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--receiver', choices=['off', 'nebius'], default='off')
    parser.add_argument('--receiver-limit', type=int, default=10)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    if args.projector == 'laptop' and not (args.training_python and args.run_dir):
        parser.error('Laptop mode requires --training-python and --run-dir.')
    receiver = None
    if args.receiver == 'nebius':
        receiver = NebiusReceiver(os.environ.get('NEBIUS_API_KEY', ''), args.receiver_limit)
    run = ROOT / 'demo-runs' / uuid4().hex
    projector = ReplayProjector() if args.projector == 'replay' else LaptopProjector(
        args.training_python, args.run_dir, run / 'worker')
    if isinstance(projector, LaptopProjector):
        atexit.register(projector.close)
    service = DemoService(projector, receiver, log_dir=run / 'events')
    print('Local receipts:', service.log_dir, flush=True)
    app = build_app(service)
    app.queue(max_size=8).launch(server_name='127.0.0.1', server_port=args.port,
                               share=False, show_error=False, footer_links=[],
                               allowed_paths=[], blocked_paths=[str(ROOT / '.env')])


if __name__ == '__main__':
    main()
