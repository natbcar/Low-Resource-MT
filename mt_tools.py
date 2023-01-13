import re

NUMOBJ = re.compile(r'[0-9]+')

LANG_REGULARIZER = {
        'ht':'hat', 'hat':'hat', 'haitian':'hat',
        'fr':'fra', 'fra':'fra', 'french':'fra',
        'en':'eng', 'eng':'eng', 'english':'eng',
        'es':'spa', 'spa':'spa', 'spanish':'spa',
        'jm':'jam', 'jam':'jam', 'jamaican':'jam',
        'th':'tha', 'tha':'tha', 'thai':'tha',
        'lo':'lao', 'lao':'lao'
        }

EPITRAN_LANGS = {
        'hat':'hat-Latn-bab',
        'fra':'fra-Latn',
        'spa':'spa-Latn',
        'jam':'jam-Latn',
        'eng':'eng-Latn',
        'tha':'tha-Thai',
        'lao':'lao-Laoo-prereform'
        }
