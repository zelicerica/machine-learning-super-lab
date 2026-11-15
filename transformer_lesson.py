"""A concept lesson and local interactive view of the original Transformer."""
from pathlib import Path


def additions(render, start):
    return [{
        'id': 'transformer-world', 'number': start,
        'title': 'The Transformer turning point',
        'description': 'Explore the architecture in 3D, then trace the ideas from attention to modern AI.',
        'topic': 'Landmark research', 'track': 'Modern language engineering',
        'kind': 'concept', 'minutes': 30,
        'scope': 'Interactive architecture and historical evidence; no model training',
        'interactive': 'transformer', 'commands': [],
        'html': render((Path(__file__).parent / 'course/Transformer-Turning-Point.md').read_text(encoding='utf-8')),
        'quiz': {
            'question': 'What did the 2017 Transformer paper introduce?',
            'options': [
                'An attention-based encoder–decoder without recurrence or convolution in its sequence architecture',
                'The first use of attention in neural networks',
                'Instruction-following chat trained with human preferences'],
            'correct': 0,
            'explanation': 'Attention predates the paper. The Transformer changed the sequence architecture; instruction tuning and preference learning are later developments.'
        }
    }]
