import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from warnings import warn

import requests
from littleutils import retry



def warn_with_ignore(message, *args, **kwargs):
    return warn(
        message +
        '\nSet the environment variable OUTDATED_IGNORE=1 to disable these warnings.',
        *args,
        **kwargs
    )


def format_date(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def cache_is_valid(cache_dt):
    return format_date(datetime.now() - timedelta(days=1)) < cache_dt


# noinspection PyCompatibility
@retry()
def get_url(url):
    return requests.get(url).text


def get_cache_filename(package):
    return 'outdated_cache_' + package


@contextmanager
def exception_to_warning(description, category, always_raise=False):
    """
    Catches any exceptions that happen in the corresponding with block
    and instead emits a warning of the given category,
    unless always_raise is True or the environment variable
    OUTDATED_RAISE_EXCEPTION is set to 1, in which caise the exception
    will not be caught.
    """

    try:
        yield
    except Exception:
        # We check for the presence of various globals because we may be seeing the death
        # of the process if this is in a background thread, during which globals
        # get 'cleaned up' and set to None
        if always_raise or os and os.environ and os.environ.get('OUTDATED_RAISE_EXCEPTION') == '1':
            raise

        if warn_with_ignore:
            warn_with_ignore(
                'Failed to %s.\n'
                'Set the environment variable OUTDATED_RAISE_EXCEPTION=1 for a full traceback.'
                % description,
                category,
            )


def constantly(x):
    return lambda *_, **__: x


class DummyFile(object):
    """
    File-like object that does nothing. All methods take any arguments
    and return an empty string.
    """

    write = read = close = __enter__ = __exit__ = constantly('')
