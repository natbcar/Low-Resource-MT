import html
import re
import string
import sys
import unicodedata
from sacremoses import MosesDetokenizer

md = {"az": MosesDetokenizer(lang="az"), "en": MosesDetokenizer(lang="en")}

MIN_WORDS = 3
MAX_WORDS = 100
MAX_CHARS = 1000
NON_TEXT_RATIO = 0.3
TEXT_RATIO = 0.7


# 1.  Detect and fix technical issues in the content
#       I accomplished this step by simply using Python's xml.etree module to parse the XML, and
#       then for each <tuv> tag, joining together all of the text within it, and then running that
#       code through a function which uses the replaces all instances of the regex [\r\n] with "".
# 2.  Remove empty segments (source or target)
#       I actually have this running later in the pipeline, where it checks the length of lines
#       after processing them further, and simply doesn't print them to the output file if either
#       the source or target has a length of 0.
# 3.  Normalize escaped characters/entities (pay attention to the “Note:”)
#       This is mostly automatically done by the XML parser. However, some still slipped through,
#       so I also wrote the normalize_entities function which removes all \ characters (some of the
#       escaped entities had a \ before the # sign) and then uses Python's html.unescape() function
#       to replace the entities.
# 4.  Normalize certain control characters/Normalize whitespaces
#       For these steps, I wrote the functions remove_control_characters (which checks if the
#       character falls in a Unicode category starting with C and removes it if it does) and
#       normalize_whitespace, which checks for any instance of whitespace using the simple regex
#       "\s+" and replaces it with a single space. These functions are run after the remove_tags
#       function described below.
# 5.  Normalize quotes (? – do this unless you’re sure all quotes are consistent)
#       For this step, I manually checked what quote characters appear in the tmx files, and then
#       wrote the normalize_quotes function which replaces single quote-like characters with single
#       quotes and double quote-likes with double quotes, matching with the regexes [‘’′] and [“”]
#       respectively.
# 6.  Removing tags that don’t affect the meaning
#       For this, I created the remove_tags function which searches for tags in curly brackets {}
#       and greater than/less than signs <>. Anything within these brackets is removed, along with
#       any extra curly brackets (many mismatched brackets remained after removing tags) and odd
#       occurences of "DOCTYPE html>" which appeared in some files.
# 7.  Identify and remove duplicates with no context for MT training purposes
#       To remove duplicates, I store all sentence pairs together in a set of tuples (unique_pairs)
#       and skip adding the pair to the output if a pair is already present in that set.
# 8.  Check if a segment contains mostly non-text content
#       To check this, I counted matches of the regex "\W" (non-word character) in each string,
#       divided by the total length of the string, and skipped the pair if either resulted in a
#       ratio greater than 0.4.
# 9. Characters that do not match either the expected source or target language(? - only if you
#     have lists of valid characters to use)
#       I did not do this step, as I can't really identify what belongs or doesn't belong to the
#       source and target languages.
# 10. Do not remove segments where source = target(? – you probably should remove them)
#       I did remove them, simply checking if source = target after cleaning strings.
# 11. Check unbalanced brackets(? – you should probably remove these, too)
#       Because there was a fairly low number of lines with unbalanced brackets, I chose to simply
#       remove lines which had an inequal number of opening and closing parantheses.
# 12. Remove entries consisting of only punctuation, whitespace, or tags (like #8)
#       This is automatically handled by the way I'm handling #8
# 13. Remove segments that are too long (>100 words)
#       For this and the following step, I wrote a word_count function which splits a string  by
#       spaces, then strips each split string of any punctuation and adds up how many of those
#       strings contain alphabetical characters.
# 14. Remove segments that are too short(<3 words)
#       See above.
# 15. Misalignments (manual check)
#       I can't check all the lines of course, but it seems that everything is aligned. The number
#       of strings matches for each language, and the beginning and end strings seem to match. I
#       did find a place in one of the TMX files where the translations were completely incorrect,
#       but as far as I can tell, that was taken care of by the sentence ratio threshold of 0.5
#       that I describe below.
# 16. Check sentence length ratios and remove if the ratio exceeds your threshold
#       For this step, I compared the length of the shorter string to the length of the longer
#       string and removed it if it was less than half the length of the long string. This loses
#       a couple of legitimate strings, but mostly cleans out places where the target is completely
#       different from the source.


def filter_chars(s, fun):
    return "".join(ch for ch in s if fun(ch))


def remove_tags(s):
    result = re.sub(r"\$?\{.+?\}|DOCTYPE html>|<.*?>", "", s)
    return re.sub(r"[\{\}]", "", result)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


def normalize_whitespace(s):
    return re.sub(r"\s+", " ", s)


def normalize_quotes(s):
    result = re.sub(r"[“”]", '"', re.sub(r"[‘’′]", "'", s))
    result = re.sub(r"« ?", '"', re.sub(r" ?»", '"', result))
    return result


def replace_misc(s):
    return re.sub(r"(\d)en(\d)", r"\1–\2", s)


def normalize_entities(s):
    result = s.replace("\\", "")
    return html.unescape(result)


def normalize_contractions(s):
    return s.replace("n 't ", "n't ")


def clean_line(tag, lang):
    result = tag.replace("\r", "")
    result = remove_tags(result)
    result = remove_control_characters(result)
    result = normalize_whitespace(result)
    result = normalize_quotes(result)
    result = normalize_entities(result)
    result = replace_misc(result)
    result = result.strip()
    result = md[lang].detokenize(result.split(" "))
    result = normalize_contractions(result)
    return result


def word_count(text):
    return sum([w.strip(string.punctuation).isalpha() for w in text.split()])


def non_text_ratio(s):
    return len(re.findall(r"[\W\d]", s)) / len(s)


def text_ratio(s):
    return len(
        re.findall(
            r"[AaBbCcÇçDdEeƏəFfGgĞğHhXxIıİiJjKkQqLlMmNnOoÖöPpRrSsŞşTtUuÜüVvYyZz]", s
        )
    ) / len(s)


def has_spaced_entities(string):
    return re.search(r"& (amp|quot|lt|gt|apos|# \d+) ?;", string)


output_en = open("tr-az/en-az.en", "w")
output_az = open("tr-az/en-az.az", "w")

unique_pairs = set()

for text_az, text_en in zip(open(sys.argv[1]), open(sys.argv[2])):
    text_az = clean_line(text_az, "az")
    text_en = clean_line(text_en, "en")
    if (
        # Remove lines with empty strings source or target
        len(text_en) == 0
        or len(text_az) == 0
        # Remove anything with fewer or greater than certain thresholds of words
        or word_count(text_en) < MIN_WORDS
        or word_count(text_az) < MIN_WORDS
        or word_count(text_en) > MAX_WORDS
        or word_count(text_az) > MAX_WORDS
        or len(text_en) > MAX_CHARS
        or len(text_az) > MAX_CHARS
        # Remove duplicates
        or (text_en, text_az) in unique_pairs
        # Remove lines with a high portion of non-word characters
        or text_ratio(text_en) < TEXT_RATIO
        or text_ratio(text_az) < TEXT_RATIO
        # Remove lines where source = target
        or text_en == text_az
        # Remove lines which have a source or target twice as long or longer than its counterpart
        or min(len(text_en), len(text_az)) / max(len(text_en), len(text_az)) < 0.5
        # Remove lines which have unbalanced brackets
        or text_en.count("(") != text_en.count(")")
        or text_az.count("(") != text_az.count(")")
        # Remove lines with weird html entities with spaces
        or has_spaced_entities(text_en)
        or has_spaced_entities(text_az)
    ):
        continue

    unique_pairs.add((text_en, text_az))

    print(text_en, file=output_en)
    print(text_az, file=output_az)
