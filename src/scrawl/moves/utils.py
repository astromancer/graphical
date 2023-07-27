
# std
from collections import abc, defaultdict

# third-party
import more_itertools as mit
from loguru import logger
from matplotlib.artist import Artist

# local
from recipes import op, pprint


def filter_non_artist(objects):
    if objects is None:
        return

    for o in filter(None, mit.collapse(objects)):
        if isinstance(o, Artist):
            yield o
            continue

        # warn if not art
        logger.warning('Object {!r} is not a matplotlib Artist. Filtering.', o)


def art_summary(artists):

    if artists is None:
        return ''

    if isinstance(artists, abc.Collection):
        col = defaultdict(list)
        for art in artists:
            col[type(art)].append(art)

        return pprint.pformat(col, '',
                              lhs=op.attrgetter('__name__'),
                              rhs=lambda l: '\n'.join(map(str, l)))

    return str(artists)
